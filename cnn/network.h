#pragma once
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <map>
#include <set>

#include "cnn/util/util.h"
#include "cnn/layers/layers.h"
#include "cnn/lossfunctions/loss_function.h"
#include "cnn/activations/activation_function.h"

namespace cnn {
    struct Result
    {
        Result(): num_success(0), num_total(0) {}

        real_t accuracy() const 
        {
            return num_success * 100.0 / num_total;
        }

        template <typename Char,typename CharTraits>
        void printSummary(std::basic_ostream<Char, CharTraits>& os) const
        {
            os << "accuracy:" << accuracy() << "% (" << num_success << "/" << num_total << ")" << std::endl;
        }

        template <typename Char,typename CharTraits>
        void printDetail(std::basic_ostream<Char, CharTraits>& os) const
        {
            printSummary(os);
            auto all_labels = labels();
            os << std::setw(5) << "*" << " ";
            for (auto c : all_labels) 
                os << std::setw(5) << c << " ";
                os << std::endl;

            for (auto r : all_labels) {
                os << std::setw(5) << r << " ";           
                for (auto c : all_labels) 
                    os << std::setw(5) << confusion_matrix[r][c] << " ";
                os << std::endl;
            }
        }

        std::set<label_t> labels() const
        {
            std::set<label_t> all_labels;
            for(auto r : confusion_matrix)
            {
                all_labels.insert(r.first);
            }
        }

        int num_success;
        int num_total;
        std::map<label_t, std::map<label_t, int> > confusion_matrix;
    };

    enum grad_check_mode {
        GRAD_CHECK_ALL, 
        GRAD_CHECK_RANDOM 
    };
    template<typename LossFunction,typename Op>
    class Network
    {
    public:
        explicit Network(const std::string& name = "") : name_(name) {};
        
        size_t in_dim() const { return layers_.head()->in_size(); }

        size_t out_dim() const { return layers_.tail()->out_size(); }

        std::string  name() const { return name_; }
        
        Optimizer&   optimizer() { return optimizer_; }

        void initWeights() { layers_.initWeights(); }


        void add(std::shared_ptr<layer_base> layer) { layers_.add(layer); }

        vec_t predict(const vec_t& in) { return fprop(in); }

        real_t predictMaxValue(const vec_t& in) {
            return fpropMax(in);
        }

        label_t predictLabel(const vec_t& in) {
                return fpropMaxIndex(in);
        }

        template <typename Range>
        vec_t predict(const Range& in) {
            using std::begin; // for ADL
            using std::end;
            return predict(vec_t(begin(in), end(in)));
        }

        template<typename OnBatchEnumerate, typename OnEpochEnumerate,typename T>
        bool train(const std::vector<vec_t>& in,const std::vector<T>& t,size_t batch_size,int epoch,OnEpochEnumerate on_batch_enumerate,OnEpochEnumerate on_epoch_enumerate,
            const bool reset_weights = true,
            const int n_threads = CNN_TASK_SIZE, 
            const std::std::vector<vec_t>* t_cost = nullptr)
        {
            checkTrainingData(in,t);
            checkTargetCostMatrix(t,t_cost);
            setNetPhase(net_phase::TRAIN);
            layers_.setWorkerCount(n_threads);
            if(reset_weights)
                initWeights();
            layers_->setParallelize(batch_size < CNN_TASK_SIZE);
            optimizer_.reset();

            for(int iter = 0; iter < epoch; iter++)
                if(optimizer_.requireHessian())
                    computeHessian();
                for(size_t i = 0; i < in.size(); i += batch_size)
                {
                    trainOnce(&in[i],&t[i],
                        static_cast<int>(std::min(batch_size,in.size() - i)),
                        n_threads,
                        geTargetCostSamplePointer(t_cost,i));
                    on_batch_enumerate();

                    if (i % 100 == 0 && layers_.isExploded()) {
                            std::cout << "[Warning]Detected infinite value in weight. stop learning." << std::endl;
                            return false;
                    }
                    on_epoch_enumerate();
                }
                return true;
        }

        template<typename T>
        bool train(const std::vector<vec_t>& in, const std::vector<T>& t, size_t batch_size = 1, int epoch = 1) {
            setNetphase(net_phase::TRAIN);
            return train(in, t, batch_size, epoch, nop, nop);
        }

        void setNetphase(net_phase phase)
        {
            for (size_t i = 0; i != layers_.depth(); ++i) {
                layers_[i]->setContext(phase);
            }
        }

        Result test(const std::vector<vec_t>& in, const std::vector<label_t>& t) {
            Result test_result;
            setNetphase(net_phase::TEST);
            for (size_t i = 0; i < in.size(); i++) {
                const label_t predicted = fpropMaxIndex(in[i]);
                const label_t actual = t[i];

                if (predicted == actual) test_result.num_success++;
                test_result.num_total++;
                test_result.confusion_matrix[predicted][actual]++;
            }
            return test_result;
        }

        std::vector<vec_t> test(const std::vector<vec_t>& in)
        {
                std::vector<vec_t> test_result(in.size());
                setNetphase(net_phase::Test);
                for_i(in.size(), [&](int i)
                {
                    test_result[i] = predict(in[i]);
                });
                return test_result;
        }

        real_t getLoss(const std::vector<vec_t>& in, const std::vector<vec_t>& t) {
            real_t sum_loss = real_t(0);

            for (size_t i = 0; i < in.size(); i++) {
                const vec_t predicted = predict(in[i]);
                sum_loss += getLoss(predict(in[i]), t[i]);
            }
            return sum_loss;
        }

        void save(std::ostream& os) const {
            os.precision(std::numeric_limits<cnn::real_t>::digits10);

            auto layer = layers_.head();
            while (layer) { layer->save(os); layer = layer->next(); }
        }

        void load(std::istream& is) {
            is.precision(std::numeric_limits<cnn::real_t>::digits10);

            auto layer = layers_.head();
            while (layer) { layer->load(is); layer = layer->next(); }
        }
    
        void fastLoad(const char* filepath) {
            FILE* stream = fopen(filepath, "r");
            std::vector<double> data;
            double temp;
            while (fscanf(stream, "%lf", &temp) > 0)
                data.push_back(temp);
            fclose(stream);

            auto layer = layers_.head();
            int idx = 0;
            while (layer) {
                layer->load(data, idx);
                layer = layer->next();
            }
        }

         bool checkGradient(const vec_t* in, const label_t* t, int data_size, real_t eps, grad_check_mode mode) {
            assert(!layers_.empty());
            std::vector<vec_t> v;
            label2vector(t, data_size, &v);

            auto current = layers_.head();

            while ((current = current->next()) != 0) { // ignore first input layer
                vec_t& w = current->weight();
                vec_t& b = current->bias();
                vec_t& dw = current->weight_diff(0);
                vec_t& db = current->bias_diff(0);

                if (w.empty()) continue;
                
                switch (mode) {
                case GRAD_CHECK_ALL:
                    for (int i = 0; i < (int)w.size(); i++)
                        if (!computeDelta(in, &v[0], data_size, w, dw, i, eps)) return false;
                    for (int i = 0; i < (int)b.size(); i++)
                        if (!computeDelta(in, &v[0], data_size, b, db, i, eps)) return false;
                    break;
                case GRAD_CHECK_RANDOM:
                    for (int i = 0; i < 10; i++)
                        if (!computeDelta(in, &v[0], data_size, w, dw, uniform_idx(w), eps)) return false;
                    for (int i = 0; i < 10; i++)
                        if (!computeDelta(in, &v[0], data_size, b, db, uniform_idx(b), eps)) return false;
                    break;
                default:
                    throw cnnError("unknown grad-check type");
                }
            }
            return true;
        }
        void check_t(size_t i, label_t t, size_t dim_out) {
            if (t >= dim_out) {
                std::ostringstream os;
                os << format_str("t[%u]=%u, dim(network output)=%u", i, t, dim_out) << std::endl;
                os << "in classification task, dim(network output) must be greater than max class id." << std::endl;
                if (dim_out == 1)
                    os << std::endl << "(for regression, use vector<vec_t> instead of vector<label_t> for training signal)" << std::endl;

                throw cnnError("output dimension mismatch!\n " + os.str());
            }
        }

        template <typename L, typename O>
        bool hasSameWeights(const Network<L, O>& others, real_t eps) const {
            auto h1 = layers_.head();
            auto h2 = others.layers_.head();

            while (h1 && h2) {
                if (!h1->hasSameWeights(*h2, eps))
                    return false;
                h1 = h1->next();
                h2 = h2->next();
            }
            return true;
        }

        const LayerBase* operator [] (size_t index) const {
        return layers_[index];
        }

        LayerBase* operator [] (size_t index) {
            return layers_[index];
        }

        size_t depth() const {
            return layers_.depth();
        }

        Index3d<size_t> in_shape() const {
            return layers_.head()->in_shape();
        }

        template <typename WeightInit>
        network& weightInit(const WeightInit& f) {
            auto ptr = std::make_shared<WeightInit>(f);
            for (size_t i = 0; i < depth(); i++)
              layers_[i]->weightInit(ptr);
            return *this;
        }

        template <typename BiasInit>
        network& biasInit(const BiasInit& f) { 
            auto ptr = std::make_shared<BiasInit>(f);
            for (size_t i = 0; i < depth(); i++)
                layers_[i]->biasInit(ptr);
            return *this;
        }

    protected:
        real_t fprop_max(const vec_t& in, int idx = 0) {
            const vec_t& prediction = fprop(in, idx);
            return *std::max_element(std::begin(prediction), std::end(prediction));
        }

        label_t fprop_max_index(const vec_t& in, int idx = 0) {
            return label_t(max_index(fprop(in, idx)));
        }

    private:
        void label2vector(const label_t* t, int num, std::vector<vec_t> *vec) const {
            cnn_size_t outdim = out_dim();

            assert(num > 0);
            assert(outdim > 0);

            vec->reserve(num);
            for (int i = 0; i < num; i++) {
                assert(t[i] < outdim);
                vec->emplace_back(outdim, target_value_min());
                vec->back()[t[i]] = target_value_max();
            }
        }

        void trainOnce(const vec_t* in, const label_t* t, int size, const int n_threads, const vec_t* t_cost) {
            std::vector<vec_t> v;
            label2vector(t, size, &v);
            trainOnce(in, &v[0], size, n_threads, t_cost);
        }

        void trainOnce(const vec_t* in, const vec_t* t, int size, const int n_threads, const vec_t* t_cost) {
            if (size == 1) {
                bprop(fprop(in[0]), t[0], 0, t_cost);
                layers_.updateWeights(&optimizer_, 1, 1);
            } else {
                train_onebatch(in, t, size, n_threads, t_cost);
            }
        }  

        void trainOneBatch(const vec_t* in,const vec_t* t,int batch_size,const int num_tasks,const vec_t* t_cost)
        {
            int num_threads = std::min(batch_size + num_threads - 1) / num_threads;

            int data_per_thread = (batch_size + num_threads - 1) / num_threads;

            for_i(num_threads,[&](int i) {
                int start_index = i * data_per_thread;
                int end_index = std::min(batch_size,start_index + data_per_thread);
                for(int j = start_index; j < end_index; ++j)
                    bprop(fprop(in[j],i),t[j],i,t_cost ? &(t_cost[j]) : nullptr);
            },1);

            layers_.updateWeights(&optimizer_,num_threads,batch_size);
        }

        void computeHessian(const std::vector<vec_t>& in,const std::vector<vec_t>* t_cost,int size_init_hessian = 500)
        {
            int size = std::min((int)in.size(),size_init_hessian);

            for(int i = 0; i < size; i++)
                bprop_2nd(fprop(in[i]),geTargetCostSamplePointer(t_cost,i));
            layers_.divideHessian(size);
        }

        template<typename Activation>
        bool isCanonicalLink(const Activation& h)
        {
            if (typeid(h) == typeid(activation::sigmoid) && typeid(E) == typeid(cross_entropy)) return true;
            if (typeid(h) == typeid(activation::tan_h) && typeid(E) == typeid(cross_entropy)) return true;
            if (typeid(h) == typeid(activation::identity) && typeid(E) == typeid(mse)) return true;
            if (typeid(h) == typeid(activation::softmax) && typeid(E) == typeid(cross_entropy_multiclass)) return true;
            return false;
        }

        const vec_t& fprop(const vec_t& in,int idx = 0)
        {
            if(in.size() != (size_t)in_dim())
                dataMismatch(*layers_[0],in);
            return layers_.head()->forward(in,idx);
        }

        real_t getLoss(const vec_t& out,const vec_t& t)
        {
            real_t e = real_t(0);
            assert(out.size() == t.size());
            for(size_t i = 0; i < out.size(); i++)
            {
                e += E::f(out[i],t[i]);
            }
            return e;
        }

        void bprop_2nd(const vec_t& out,const vec_t* t_cost)
        {
            vec_t delta(out_dim());
            const Activation::function& h = layers_.tail()->activation_function();
            if(isCanonicalLink(h))
            {
                for_i(out_dim(),[&](int i)
                {
                    delta[i] = target_value_max() * h.df(out[i]);
                });
            }
            else
            {
                for_i(out_dim(),[&](int i)
                {
                    delta[i] = target_value_max() * h.df(out[i]) * h.df(out[i]);
                });
            }

            if(t_cost)
            {
                for_i(out_dim(),[&](int i )
                {
                    delta[i] *= (*t_cost)[i];
                });
            }

            layers_.tail()->backward_2nd(delta);
        }

        void bprop(const vec_t& out,const vec_t& t,int idx,const vec_t* t_cost)
        {
            vec_t delta(out_dim());
            const Activation::function& h = layers_.tail()->activation_function();
            if(isCanonicalLink(h))
            {
                for_i(out_dim(),[&](int i)
                {
                    delta[i] = out[i] - t[i];
                });
            }
            else
            {
                vec_t dEdy = Gradient<E>(out,t);

                for(size_t i = 0; i < out_dim(); i++)
                {
                    vec_t dyda = h.df(out,i);
                    delta[i] = vectorize::dot(&dEdy[0],&dyda[0],out_dim());
                }
            }

            if(t_cost)
            {
                for_i(out_dim(),[&](int i )
                {
                    delta[i] *= (*t_cost)[i];
                });
            }

            layers_.tail()->backward(delta);
        }

        bool computeDelta(const vec_t* in ,const vec_t* v,int data_size,vec_t& w,vec_t& dw,int check_index,double eps)
        {
            static const real_t delta = 1e-10;
            std::fill(dw.begin(),dw.end(),real_t(0));

            real_t prev_w = w[check_index];
            w[check_index] = prev_w + delta;
            real_t f_p = real_t(0);
            for(int i = 0; i < data_size; i++)
            {
                f_p += getLoss(fprop(in[i],v[i]));
            }

            real_t f_m = real_t(0);
            w[check_index] = prev_w - delta;
            for(int i = 0; i < data_size; i++)
            {
                f_p += getLoss(fprop(in[i],v[i]));
            }

            real_t delta_by_numerical = (f_p - f_m) / (real_t(2) * delta);
            w[check_index] = prev_w;

            for(int  i = 0; i < data_size; i++)
            {
                bprop(fprop(in[i]),v[i],0,nullptr);
            }

            real_t delta_by_bprop = dw[check_index];
            return std::abs(delta_by_bprop - delta_by_numerical) < eps;
        }

        
        template <typename T>
        const T& at(size_t index) const {
            return layers_.at<T>(index);
        }
        void check_t(size_t i, const vec_t& t, size_t dim_out) {
            if (t.size() != dim_out)
                throw cnnError(format_str("output dimension mismatch!\n dim(target[%u])=%u, dim(network output size=%u", i, t.size(), dim_out));
        }

        template<typename T>
        void checkTrainingData(const std::std::vector<vec_t> in,const std::vector<T> t)
        {
            size_t dim_in = in_dim();
            size_t dim_out = out_dim();

            if(in.size() != t.size())
                throw cnnError("number of training data must be equal to label data");

            for (size_t i = 0; i < num; i++) {
                if (in[i].size() != dim_in)
                    throw cnnError(format_str("input dimension mismatch!\n dim(data[%u])=%d, dim(network input)=%u", i, in[i].size(), dim_in));

                check_t(i, t[i], dim_out);
            }
        }

        typename<typename T>
        void checkTargetCostMatrix(const std::vector<T>& t, const std::vector<vec_t>* t_cost) {
            if (t_cost != nullptr) {
                if (t.size() != t_cost->size()) {
                    throw cnnError("if target cost is supplied, its length must equal that of target data");
                    }

            for (size_t i = 0, end = t.size(); i < end; i++) {
                checkTargetCostElement(t[i], t_cost->operator[](i));
                }
            }
        }

         void checkTargetCostElement(const label_t t, const vec_t& t_cost) {
            if (t >= t_cost.size()) {
                throw cnnError("if target cost is supplied for a classification task, some cost must be given for each distinct class label");
            }
        }


        void checkTargetCostElement(const vec_t& t, const vec_t& t_cost) {
            if (t.size() != t_cost.size()) {
                throw cnnError("if target cost is supplied for a regression task, its shape must be identical to the target data");
            }
        }
        inline const vec_t* geTargetCostSamplePointer(const std::vector<vec_t>* t_cost, size_t i) {
            if (t_cost) {
                const std::vector<vec_t>& target_cost = *t_cost;
                assert(i < target_cost.size());
                return &(target_cost[i]);
            }
            else {
                return nullptr;
            }
        }
        
         float_t target_value_min() const { return layers_.tail()->activation_function().scale().first; }
        float_t target_value_max() const { return layers_.tail()->activation_function().scale().second; }

        std::string name_;
        Optimizer optimizer_;
        Layers layers_;
    };
}
#ifndef __MRFENERGY_H__
#define __MRFENERGY_H__

#include "instances.h"

template <class T> class MRFEnergy
{
private:    
    struct Node;

public:
    typedef typename T::Label Label;
    typedef typename T::Real Real;
    typedef typename T::GlobalSize GlobalSize;
    typedef typename T::LocalSize LocalSize;
    typedef typename T::NodeData NodeData;
    typedef typename T::EdgeData EdgeData;

    typedef Node* NodeId;
    typedef void (*ErorrFunction)(char* msg);

    MRFEnergy(GlobalSize kglobal,ErorrFunction err_fn);

    ~MRFEnergy();

    NodeId addNode(LocalSize k,NodeData data);

    void addNodeData(NodeId i,NodeData data);

    void addEdge(NodeId i,NodeId j,EdgeData data);

    void clearMessages();

    void addRandomMessages(unsigned int random_seed,Real min_value,Real max_value);

    void setAutomaticOrdering();

    struct Options
    {
        Options()
        {
            eps_ = -1;
            iter_max_ = 1000000;
            print_iter_ = 5;
            print_min_iter_ = 10;
        }

        Real eps_;
        int iter_max_;
        int print_iter_;
        int print_min_iter_;
    };

    int minimize_TRW_S(Options& options,Real& lower_bound,Real& energy,Real* min_marginals = NULL);

    int minimize_BP(Options& options,Real& energy,Real* min_marginals = NULL);

    Label getSolution(NodeId i);

private:
    typedef typename T::Vector Vector;
    typedef typename T::Edge Edge;

    struct MRFEdge;
    struct MallocBlock;

    ErorrFunction err_fn_;
    MallocBlock malloc_block_first_;
    Node* node_first_;
    Node* node_last_;
    int node_num_;
    int edge_num_;
    GlobalSize kglobal_;
    int vector_maxsize_in_bytes_;
    bool is_energy_construction_completed_;
    char* buf_;

    void completeGraphConstruction();
    void setMonotonicTrees();
    Real computeSolutionAndEnergy();

    struct Node
    {
        int ordering_;
        MRFEdge* first_forward_;
        MRFEdge* first_backward_;

        Node* prev_;
        Node* next_;

        Label solution_;
        LocalSize k_;
        Vector d_;
    };

    struct MRFEdge
    {
        MRFEdge* next_forward_;
        MRFEdge* next_backward_;

        Node* tail_;
        Node* head_;
        Real gamma_forward_;
        Real gamma_backward_;
        Edge message_;
    };

    struct MallocBlock
    {
        static const int min_blocksize_in_bytes = 4096 - 3 * sizeof(void*);
        MallocBlock* next_;
        char* current_;
        char* last_;
    };
    char* malloc(it bytes_num);
};

template <class T> inline char* MRFEnergy<T>::malloc(int bytes_num)
{
    if(!malloc_block_first_ || malloc_block_first_->current_ + bytes_num > malloc_block_first_->last_)
    {
        int size = (bytes_num > MallocBlock::min_blocksize_in_bytes) ? bytes_num : MallocBlock::min_blocksize_in_bytes;
        MallocBlock* block = (MallocBlock*) new char[sizeof(MallocBlock) + size];
        if(!block)
            err_fn_("Not enough memory");
        block->current_ = (char*)block + sizeof(MallocBlock);
        block->last_ = block->current_ + size;
        block->next_ = malloc_block_first_;
        malloc_block_first_ = block;
    }

    char* ptr = malloc_block_first_->current_;
    malloc_block_first_->current_ += bytes_num;
    return ptr;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "MRFEnergy.h"

#include "instances.inc"
void defaltErrorFn(char *msg)
{
    fprintf(stderr,"%s\n",msg);
    exit(1);
}


template<class T> MRFEnergy::MRFEnergy(GlobalSize kglobal,ErorrFunction err_fn) 
            : err_fn_(err_fn ? err_fn : defaltErrorFn),
            malloc_block_first_(NULL),
            node_first_(NULL),
            node_last_(NULL),
            node_num_(0),
            edge_num_(0),
            kglobal_(kglobal),
            vector_maxsize_in_bytes_(0),
            is_energy_construction_completed_(false),
            buf_(NULL) {}

template<class T> MRFEnergy<T>::~MRFEnergy<T>()
{
    while(malloc_block_first_)
    {
        MallocBlock* next = malloc_block_first_->next_;
        delete malloc_block_first_;
        malloc_block_first_ = next;
    }
}

template <class T> typename MRFEnergy<T>::NodeId addNode(LocalSize k,NodeData data)
{
    if(is_energy_construction_completed_)
    {
        err_fn_("Error in AddNode(): graph construction completed - nodes cannot be added");
    }

    int actual_vector_size = Vector::getSizeInBytes(kglobal_,k);
    if(actual_vector_size < 0)
    {
        err_fn_("Error in AddNode() (invalid parameter?)");
    }
    if(vector_maxsize_in_bytes_ < actual_vector_size)
    {
        vector_maxsize_in_bytes_ = actual_vector_size;
    }
    int node_size = sizeof(Node) - sizeof(Vector) + actual_vector_size;
    Node* i = (Node *)malloc(node_size);

    i->k_ = k;
    i->d_.initialize(kglobal_,k,data);

    i->first_forward_=NULL;
    i->first_backward_=NULL;
    i->prev_=node_last_;
    if(node_last_)
    {
        node_last_->next_=i;
    }
    else 
    {
        node_first_ = i;
    }
    node_last_ = i;
    i->next_ = NULL;

    i->ordering_ = node_num_++;

    return i;
}

template <class T> void MRFEnergy<T>::addNodeData(NodeId i,NodeData data)
{
    i->d_.add(kglobal_,i->k_,data);
}

template <class T> void MRFEnergy::addEdge(NodeId i,NodeId j,EdgeData data)
{
    if(is_energy_construction_completed_)
    {
        err_fn_("Error in addEdge(): graph construction completed - nodes cannot be added");
    }

    MRFEdge* e;
    int actual_edge_size = Edge::getSizeInBytes(kglobal_,i->k_,j->k_,data);
    if(actual_edge_size < 0)
    {
        err_fn_("Error in AddEdge() (invalid parameter?)");
    }
    int edge_size = sizeof(MRFEdge) - sizeof(Edge) + actual_edge_size;
    e = (MRFEdge*)malloc(edge_size);

    e->message_.initialize(kglobal_,i->k_,j->k_,data,&i->d_,&j->d_);

    e->tail_ = i;
    e->next_forward_ = i->first_forward_;
    e->first_forward_ = e;

    e->head_ = j;
    e->next_backward_ = j->first_backward_;
    j->first_backward_ = e;

    edge_num_++;
}

template <class T> void MRFEnergy<T>::clearMessages()
{
    Node* i;
    MRFEdge* e;

    if(!is_energy_construction_completed_)
    {
        completeGraphConstruction();
    }

    for(i = node_first_; i ; i = i->next_)
    {
        for(e = i->first_forward_; e; e=e->next_forward_)
        {
            e->message_.getMessagePtr()->setZero(kglobal_,i->k_);
        }
    }
}

template <class T> void MRFEnergy<T>::addRandomMessages(unsigned int random_seed,Real min_value,Real max_value)
{
    Node* i;
    MRFEdge* e;
    int k;
    if(!is_energy_construction_completed_)
    {
        completeGraphConstruction();
    }

    srand(random_seed);

    for(i = node_first_; i ; i = i->next_)
    {
        for(e = i->first_forward_; e; e=e->next_forward_)
        {
            Vector* m = e->message_.getMessagePtr();
            for(k = 0; k < m->getArraySize(kglobal_,i->k_);k++)
            {
                Real x = (Real)(min_value + random() / ((double)RAND_MAX) * (max_value - min_value));
                x+=m->getArrayValue(kglobal_,i->k_,k);
                m->setArrayValue(kglobal_,i->k_,k,x);
            }
        }
    }
}

template <class T> void MRFEnergy<T>::completeGraphConstruction()
{
    Node* i;
    Node* j;
    MRFEdge* e;
    MRFEdge* e_prev;

    if(is_energy_construction_completed_)
    {
        err_fn_("Fatal error in CompleteGraphConstruction");
    }

    printf("Completing graph construction... ");

    if (buf_)
    {
        err_fn_("CompleteGraphConstruction(): fatal error");
    }

    buf_ = (char*)malloc(vector_maxsize_in_bytes_ + (vector_maxsize_in_bytes_ > Edge::getBufSizeInBytes(vector_maxsize_in_bytes_) ? vector_maxsize_in_bytes_ : Edge::getBufSizeInBytes(vector_maxsize_in_bytes_)));

#ifdef _DEBUG
    int ordering;
    for(i = node_first_, ordering = 0; i; i = i->next_,ordering++)
    {
        if((i->ordering_ != ordering) || (i->ordering_ == 0 && i->prev_) || (i->ordering_ != 0 && i->prev_->ordering_ != ordering - 1))
        {
        err_fn_("CompleteGraphConstruction(): fatal error (wrong ordering)");
        }
    }
    if (ordering != node_num_)
    {
        err_fn_("CompleteGraphConstruction(): fatal error");
    }
#endif
    for(i = node_first_; i; i = i->next_)
    {
        i->first_backward_ = NULL;
    }
    for(i = node_first_; i; i = i->next_)
    {
        e_prev = NULL;
        for(e = i->first_forward_; e;)
        {
            assert(i==e->tail_);
            j = e->head_;
            if(i->ordering_ < j->ordering_)
            {
                e->next_backward_ = j->first_backward_;
                j->first_backward_ = e;

                e_prev = e;
                e = e->next_forward_;
            }
            else
            {
                e->message_.swap(kglobal_,i->k_,j->k_);
                e->tail_ = j;
                e->head_ = i;

                MRFEdge* e_next = e->next_forward_;

                if(e_prev)
                {
                    e_prev->next_forward_ = e->next_forward_;
                }
                else
                {
                    i->first_forward_ = e->next_forward_;
                }

                e->next_forward_ = j->first_forward_;
                j->first_forward_ = e;
                e->next_backward_ = i->first_backward_;
                i->first_backward_ = e;

                e = e_next;
            }
        }
    }

    is_energy_construction_completed_ = true;

    printf("done\n");
}

template <class T> void MRFEnergy<T>::setAutomaticOrdering()
{
    int d_min;
    Node* i;
    Node8 i_min;
    Node* list;
    Node* list_boundary;
    MRFEdge* e;

    if(is_energy_construction_completed_)
    {
        err_fn_("Error in SetAutomaticOrdering(): function cannot be called after graph construction is completed");
    }

    printf("Setting automatic ordering... ");

    list = node_first_;
    list_boundary = NULL;
    node_first_ = node_last_ = NULL;
    for(i = list; i; i=i->next_)
    {
        i->ordering_ = 2 * node_num_;

        for(e = i->first_forward_; e; e=e->next_forward_)
        {
            i->ordering_++;
        }
        for(e = i->first_backward_; e; e=e->next_backward_)
        {
            i->ordering_++;
        }
    }

    while(list)
    {
        d_min = node_num_;
        for(i = list; i; i = i->next_)
        {
            assert(i->ordering_ >= 2 * node_num_);
            if(d_min > i->ordering_ - 2 * node_num_)
            {
                d_min = i->ordering_ - 2 * node_num_;
                i_min = i;
            }
        }
        i = i_min;

        if(i->prev_)
            i->prev_->next_ = i->next_;
        else
        list = i->next_; 
        if(i->next_)
            i->next_->prev_ = i->prev_;

        list_boundary = i;
        i->prev_ = NULL;
        i->next_ = NULL;

        while(list_boundary)
        {
            d_min = node_num_;
            for(i = list_boundary; i; i = i->next_)
            {
                assert(i->ordering_ >= node_num_ && i->ordering_ < 2 * node_num_);
                if(d_min > i->ordering_ - node_num_)
                {
                    d_min = i->ordering_ - node_num_;
                    i_min = i;
                }
            }
            i = i_min;

            if(i->prev_)
            i->prev_->next_ = i->next_;
            else
            list = i->next_; 
            if(i->next_)
                i->next_->prev_ = i->prev_;

            if(node_last_)
            {
                node_last_->next_ = i;
                i->ordering_ = node_last_->ordering_ + 1;
            }
            else
            {
                node_first_ = i;
                i->ordering_ = 0;
            }
            i->prev_ = node_last_;
            node_last_ = i;
            i->next_ = NULL;

            for(e = node_last_->first_forward_; e; e = e->next_forward_)
            {
                assert(node_last_ == e->tail_);
                i = e->head_;
                if(i->ordering_ >= node_num_)
                {
                    i->ordering_--;
                    if(i->ordering_ >= 2 * node_num_)
                    {
                        if(i->prev_)
                            i->prev_->next_ = i->next_;
                        else
                        list = i->next_; 
                        if(i->next_)
                            i->next_->prev_ = i->prev_;

                        list_boundary = i;
                        i->prev_ = NULL;
                        i->next_ = list_boundary;
                        list_boundary = i;
                        i->ordering_ -= node_num_; 
                    }
                }
            }
            for(e = node_last_->first_backward_; e; e=e->next_backward_)
            {
                assert(node_last_ == e->head_);
                i = e->tail_;
                if(i->ordering_ >= node_num_)
                {
                    i->ordering_--;
                    if(i->ordering_ >= 2 * node_num_)
                    {
                        if(i->prev_)
                            i->prev_->next_ = i->next_;
                        else
                        list = i->next_; 
                        if(i->next_)
                            i->next_->prev_ = i->prev_;

                        list_boundary = i;
                        i->prev_ = NULL;
                        i->next_ = list_boundary;
                        list_boundary = i;
                        i->ordering_ -= node_num_; 
                    }
                }
            }
        }

        printf("done\n");
        completeGraphConstruction();
    }
}



template <class T> void MRFEnergy<T>::setMonotonicTrees()
{
    Node* i;
    MRFEdge* e;
    if(!is_energy_construction_completed_)
    {
        completeGraphConstruction();
    }

    for(i = node_first_; i; i = i ->next_)
    {
        Real mu;
         int n_forward = 0,n_backward = 0;
         for(e = i->first_forward_; e; e = e->next_forward_)
         {
            n_forward++;
         }
         for(e = i->first_backward_; e; e = e->next_backward_)
         {
            n_backward++;
         }
         int ni = (n_forward > n_backward) ? n_forward : n_backward;

         mu = (Real)1 / ni;
         for(e = i->first_forward_; e; e = e->next_forward_)
         {
            e->gamma_forward_ = mu;
         }
         for(e = i->first_backward_; e; e = e->next_backward_)
         {
            e->gamma_backward_ = mu;
         }
    }
}

template <class T> int MRFEnergy<T>::minimize_TRW_S(Options& options,Real& lower_bound,Real& energy,Real* min_marginals)
{
    Node* i;
    Node* j;
    MRFEdge* e;
    Real v_min;
    int iter;
    Real lower_bound_prev;

    if(!is_energy_construction_completed_)
    {
        completeGraphConstruction();
    }

    printf("TRW_S algorithm\n");

    setMonotonicTrees();

    Vector* di = (Vector*)buf_;
    void* buf = (void*)(buf_ + vector_maxsize_in_bytes_);

    iter = 0;
    bool last_iter = false;

    for(iter = 1; ; iter++)
    {
        if(iter >= options.iter_max_)
            last_iter = true;
        Real* min_marginals_ptr = min_marginals;

        for(i = node_first_; i; i = i->next_)
        {
            di->copy(kglobal_,i->k_, &i->d_);
            
        }
    }
}
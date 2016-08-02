#pragma once
#include <set>
#include <stack>
#include <deque>
#include <algorithm>

template<typename stored_type,typename tree_type,typename set_container_type>
class basic_tree
{
public:
    typedef basic_tree<stored_type,tree_type,set_container_type> basic_tree_type;
    typedef stored_type* (*tclone_fcn) (const stored_type&);
private:
    struct pre_order_iterator_impl;
    struct post_order_iterator_impl;
    struct level_order_iterator_impl;

public:
    basic_tree(): parent_node(NULL), data(NULL) {}
    basic_tree(const stored_type& stored_obj);
    virtual ~basic_tree();

public:
    class const_iterator : public std::iterator<std::bidirectional_iterator_tag,
    stored_type>
    {
    public:
        typedef class basic_tree<stored_type,tree_type,set_container_type> basic_tree_type;
        const_iterator() {}
        const_iterator(typename set_container_type::const_iterator it_) : it(it_) {}
        virtual ~const_iterator() {}

        friend bool operator != (const const_iterator& lhs,const const_iterator& rhs) { return lhs.it != rhs.it; }
        friend bool operator == (const const_iterator& lhs,const const_iterator& rhs) { return lhs.it == rhs.it; }

        const tree_type& operator*() const { return it.operator *(); }
        const tree_type* operator->() const { return it.operator ->(); }

        const_iterator& operator ++() { ++it; return *this; }
        const_iterator iterator ++(int) { const_iterator old(*this); ++*this; return old; }
        const_iterator& operator --() { --it; return *this; }
        const_iterator iterator --(int) { const_iterator old(*this); --*this; return old; }

        const tree_type* node() { return *it; }
        friend class post_order_iterator;
        friend class pre_order_iterator;
        friend class level_order_iterator;

    protected:
        typename set_container_type::const_iterator it;
    };

    friend class const_iterator;

    class iterator : const_iterator
    {
    public:
        using const_iterator::it;
        typedef class basic_tree<stored_type, tree_type, set_container_type> basic_tree_type;

        iterator() {}
        iterator(typename set_container_type::iterator it_) : const_iterator(it_) {}
        ~iterator() {}

        friend bool operator != ( const iterator& lhs, const iterator& rhs )  { return lhs.it != rhs.it; }
        friend bool operator == ( const iterator& lhs, const iterator& rhs )  { return lhs.it == rhs.it; }

        tree_type& operator*() { return it.operator *(); }
        tree_type* operator->() { return it.operator ->(); }
        iterator& operator ++() { ++it;  return *this; }
        iterator operator ++(int) { iterator old(*this); ++*this; return old; }
        iterator& operator --() { --it; return *this; }
        iterator operator --(int) { iterator old(*this); --*this; return old; }

        tree_type* node() { return *it; }
    };
    friend class iterator;

    class const_pre_oreder_iterator : public std::iterator<std::bidirectional_iterator_tag,stored_type>
    {
    public:
        const_pre_oreder_iterator() {}
        const_pre_oreder_iterator(const basic_tree_type* top_node_) { it = top_node_->children.begin();
            top_node = top_node_;}
        protected:
            explicit const_pre_oreder_iterator(const_iterator& it_) : it(it_) {}
        public:
            friend bool operator != ( const const_pre_order_iterator& lhs, const const_pre_order_iterator& rhs ) { return lhs.it != rhs.it; }
            friend bool operator == ( const const_pre_order_iterator& lhs, const const_pre_order_iterator& rhs ) { return lhs.it == rhs.it; }
            const_pre_order_iterator& operator ++() { return impl.incr(this);}
            const_pre_order_iterator operator ++(int) {
                const_pre_order_iterator old(*this);
                ++*this; return old; }
            const_pre_order_iterator operator --() { return impl.decr(this); }
            const_pre_order_iterator operator --(int) { const_pre_order_iterator old(*this);
                --*this; return old; }

            const tree_type& operator*() const { return it.operator *(); }
            const tree_type* operator->() const { return it.operator ->(); }
            const tree_type* node() { return it.node(); }
            friend struct pre_order_iterator_impl;
            friend class basic_tree<stored_type,tree_type,set_container_type>;

        protected:
            std::stack<const_iterator> node_stack;
            const basic_tree_type* top_node;
            const_iterator it;
            pre_order_iterator_impl impl;
            typename set_container_type::const_reverse_iterator rit;
    };
    friend class const_pre_order_iterator;

    class pre_order_iterator : public const_pre_order_iterator
        {
        public:
            using const_pre_order_iterator::it;
            // constructors/destructor
            pre_order_iterator() {}
            pre_order_iterator( basic_tree_type* top_node_) : const_pre_order_iterator(top_node_) {}
        protected:
            explicit pre_order_iterator(iterator& it_) : const_pre_order_iterator(it_) {}

        public:
            // overloaded operators
            friend bool operator != ( const pre_order_iterator& lhs, const pre_order_iterator& rhs ) { return lhs.it != rhs.it; }
            friend bool operator == ( const pre_order_iterator& lhs, const pre_order_iterator& rhs ) { return lhs.it == rhs.it; }
            pre_order_iterator& operator ++() { ++(*static_cast<const_pre_order_iterator*>(this)); return *this; }
            pre_order_iterator operator ++(int) { pre_order_iterator old(*this); ++*this; return old; }
            pre_order_iterator operator --() { --(*static_cast<const_pre_order_iterator*>(this)); return *this; }
            pre_order_iterator operator --(int) { pre_order_iterator old(*this); --*this; return old; }

            // public interface
            tree_type& operator*() { return  it.operator *(); }
            tree_type* operator->() { return const_cast<tree_type*>(it.operator ->()); }
            tree_type* node() { return const_cast<tree_type*>(it.node()); }
            friend class basic_tree<stored_type, tree_type, set_container_type>;
        };
        friend class pre_order_iterator;

        class const_post_order_iterator : public std::iterator<std::bidirectional_iterator_tag, stored_type>
        {
        public:
            // constructors/destructor
            const_post_order_iterator() {}
            const_post_order_iterator(const basic_tree_type* top_node_) { impl.init(this, top_node_); }
        protected:
            explicit const_post_order_iterator(const_iterator& it_) : it(it_) {}

        public:
            // overloaded operators
            friend bool operator != ( const const_post_order_iterator& lhs, const const_post_order_iterator& rhs ) { return lhs.it != rhs.it; }
            friend bool operator == ( const const_post_order_iterator& lhs, const const_post_order_iterator& rhs ) { return lhs.it == rhs.it; }
            const_post_order_iterator& operator ++() { return impl.incr(this); }
            const_post_order_iterator operator ++(int) { const_post_order_iterator old(*this); ++*this; return old; }
            const_post_order_iterator operator --() { return impl.decr(this); }
            const_post_order_iterator operator --(int) { const_post_order_iterator old(*this); --*this; return old; }

            // public interface
            const tree_type& operator*() const { return  it.operator *(); }
            const tree_type* operator->() const { return it.operator ->(); }
            const tree_type* node() { return it.node(); }
            friend struct post_order_iterator_impl;
            friend class basic_tree<stored_type, tree_type, set_container_type>;

            // data
        protected:
            std::stack<const_iterator> node_stack;   
            const basic_tree_type* pTop_node;
            const_iterator it;
            post_order_iterator_impl impl;
            typename set_container_type::const_reverse_iterator rit;
        };
        friend class const_post_order_iterator;


        class post_order_iterator : public const_post_order_iterator
        {
        public:
            using const_post_order_iterator::it;
            // constructors/destructor
            post_order_iterator() {}
            post_order_iterator(basic_tree_type* top_node_) : const_post_order_iterator(top_node_) { }
        protected:
            explicit post_order_iterator(iterator& it_) : const_post_order_iterator(it_) {}

        public:
            // overloaded operators
            friend bool operator != ( const post_order_iterator& lhs, const post_order_iterator& rhs ) { return lhs.it != rhs.it; }
            friend bool operator == ( const post_order_iterator& lhs, const post_order_iterator& rhs ) { return lhs.it == rhs.it; }
            post_order_iterator& operator ++() { ++(*static_cast<const_post_order_iterator*>(this)); return *this; }
            post_order_iterator operator ++(int) { post_order_iterator old(*this); ++*this; return old; }
            post_order_iterator operator --() { --(*static_cast<const_post_order_iterator*>(this)); return *this; }
            post_order_iterator operator --(int) { post_order_iterator old(*this); --*this; return old; }

            // public interface
            tree_type& operator*() { return  it.operator *(); }
            tree_type* operator->() { return const_cast<tree_type*>(it.operator ->()); }
            tree_type* node() { return const_cast<tree_type*>(it.node()); }
            friend class basic_tree<stored_type, tree_type, set_container_type>;

        };
        friend class post_order_iterator;



        class const_level_order_iterator : public std::iterator<std::forward_iterator_tag, stored_type>
        {
        public:
            // constructors/destructor
            const_level_order_iterator() {}
            const_level_order_iterator(const basic_tree_type* top_node_) : pTop_node(top_node_), node_depth(0) { it = top_node_->children.begin(); }
        protected:
            explicit const_level_order_iterator(const_iterator& it_) : it(it_) {}

        public:
            // overloaded operators
            friend bool operator != (const const_level_order_iterator& lhs, const const_level_order_iterator& rhs) { return lhs.it != rhs.it; }
            friend bool operator == (const const_level_order_iterator& lhs, const const_level_order_iterator& rhs) { return lhs.it == rhs.it; }
            const_level_order_iterator operator ++() { return impl.incr(this); }
            const_level_order_iterator operator ++(int) { const_level_order_iterator old(*this); ++*this; return old; }
            // declare, but don't define decr operators
            const_level_order_iterator operator --();
            const_level_order_iterator operator --(int);

            // public interface
            const tree_type& operator*() const { return  it.operator *(); }
            const tree_type* operator->() const { return it.operator ->(); }
            int depth() const { return node_depth; }
            const tree_type* node() { return it.node(); }
            friend struct level_order_iterator_impl;
            friend class basic_tree<stored_type, tree_type, set_container_type>;

            // data
        protected:
            const_iterator it;
            std::deque<const_iterator> node_deque;
            const basic_tree_type* pTop_node;
            level_order_iterator_impl impl;
            int node_depth;
        };
        friend class level_order_iterator;

        class level_order_iterator : public const_level_order_iterator
        {
        public:
            using const_level_order_iterator::it;
            // constructors/destructor
            level_order_iterator() {}
            level_order_iterator(basic_tree_type* top_node_) : const_level_order_iterator(top_node_) { }
        private:
            explicit level_order_iterator(iterator& it_) : const_level_order_iterator(it_) {}
        
        public:
            // overloaded operators
            friend bool operator != (const level_order_iterator& lhs, const level_order_iterator& rhs) { return lhs.it != rhs.it; }
            friend bool operator == (const level_order_iterator& lhs, const level_order_iterator& rhs) { return lhs.it == rhs.it; }
            level_order_iterator& operator ++() { ++(*static_cast<const_level_order_iterator*>(this)); return *this; }
            level_order_iterator operator ++(int) { level_order_iterator old(*this); ++*this; return old; }

            // public interface
            tree_type& operator*() { return  it.operator *(); }
            tree_type* operator->() { return const_cast<tree_type*>(it.operator ->()); }
            tree_type* node() { return const_cast<tree_type*>(it.node()); }
            friend class basic_tree<stored_type, tree_type, set_container_type>;

        };
        friend class level_order_iterator;

        iterator find(const stored_type& stored_obj);
        const_iterator find(const stored_type& stored_obj) const;
        bool is_root() const { parent_node == NULL; }
        tree_type* parent() const { return parent_node;}
        bool empty() const { return children.empty(); }
        int size() const { return static_cast<int>(children.size()); }

        const_iterator begin() const { return const_iterator(children.begin()); }
        const_iterator end() const { return const_iterator(children.end()); }
        iterator begin() { return iterator(children.begin()); }
        iterator end() { return iterator(children.end()); }
        post_order_iterator post_order_begin() { post_order_iterator it(this); return it; }
        post_order_iterator post_order_end() { iterator it = end(); return post_order_iterator(it); }
        const_post_order_iterator post_order_begin() const { const_post_order_iterator it(this); return it; }
        const_post_order_iterator post_order_end() const { return const_post_order_iterator(end()); }
        pre_order_iterator pre_order_begin() { pre_order_iterator it(this); return it; }
        pre_order_iterator pre_order_end() { iterator it = end(); return pre_order_iterator(it); }
        const_pre_order_iterator pre_order_begin() const { const_pre_order_iterator it(this); return it; }
        const_pre_order_iterator pre_order_end() const {  return const_pre_order_iterator(end()); }
        level_order_iterator level_order_begin() { level_order_iterator it(this); return it; }
        level_order_iterator level_order_end() { iterator it = end(); return level_order_iterator(it); }
        const_level_order_iterator level_order_begin() const { const_level_order_iterator it(this); return it; }
        const_level_order_iterator level_order_end() const { return const_level_order_iterator(end()); }

        bool erase(const stored_type& stored_obj);
        stored_type* get() const { return data; }
        void set(const stored_type& stored_obj);
        void for_each(void (*fcn)(tree_type*));
        template<typename T> void for_each( T& func_obj);
        static void set_clone(const tclone_fcn& fcn) { clone_fcn = fcn; }
    protected:
        iterator insert( const stored_type& stored_obj, tree_type* parent);
        iterator insert( stored_type* stored_obj, tree_type* parent);
        iterator insert(const tree_type& tree_obj,tree_type* parent);
        void set(const tree_type& tree_obj,tree_type* parent);

    protected:
        stored_type* data;
        mutable tree_type* parent_node;
        set_container_type children;
        static tclone_fcn clone_fcn;
    private:
        struct  pre_order_iterator_impl
        {
            template<typename T> T& incr(T* self)
            {
                if( !self->it->empty() )
                {
                    self->node_stack.push(self->it);
                    self->it = self->node()->children.begin();
                }
            }
        };
  };
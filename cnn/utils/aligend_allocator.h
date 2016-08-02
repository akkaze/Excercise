#pragma once 
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#endif

#include "cnn_error.h"

namespace cnn {
    template <typename T,std::size_t alignment>
    class AlignedAlloctor
    {
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef const T* const_pointer;

        template <typename U>
        struct rebind
        {
            typedef AlignedAlloctor<U,alignment> other;
        };

        const_pointer address(const_reference value) const
        {
            return std::addressof(value);
        }

        pointer address(reference value) const
        {
            return std::addressof(value);
        }

        pointer allocate(size_type size, const void* = nullptr) {
        void* p = alignedAlloc(alignment, sizeof(T) * size);
        if (!p && size > 0)
            throw cnnError("failed to allocate");
        return static_cast<pointer>(p);
        }
        
        size_type max_size() const {
            return ~static_cast<std::size_t>(0) / sizeof(T);
        }

        void deallocate(pointer ptr, size_type)
        {
            alignedFree(ptr);
        }

        template<class U, class V>
        void construct(U* ptr, const V& value) 
        {
            void* p = ptr;
            ::new(p) U(value);
        }

    #if defined(_MSC_VER) && _MSC_VER <= 1800
    #else
        template<class U,class... Args>
        void construct(U* ptr,Args&&.. args)
        {
            void* p = ptr;
            ::new(p) U(std::forward<Args>(args)...);
        }
    #endif
    template<class U>
    void construct(U* ptr) {
        void* p = ptr;
        ::new(p) U();
    }

    template<class U>
    void destroy(U* ptr) {
        ptr->~U();
    }

    private:
        void* aligendAlloc(size_type align,size_type size) const
        {
        #if defined(_MSC_VER)
            return ::_aligned_malloc(size,align);
        #elif defined(__ANDROID__)
            return ::memalign(align,size);
        #else
            void *p;
            if (::posix_memalign(&p,align,size) != 0)
            {
                p = 0;
            }    
            return p;
        #endif    
        }

        void alignedFree(pointer ptr)
        {
        #if defined(_MSC_VER)
            ::_aligned_free(ptr);
        #else
            ::free(ptr);
        #endif
        }

        template<typename T1, typename T2, std::size_t alignment>
        inline bool operator==(const aligned_allocator<T1, alignment>&, const aligned_allocator<T2, alignment>&)
        {
            return true;
        }

        template<typename T1, typename T2, std::size_t alignment>
        inline bool operator!=(const aligned_allocator<T1, alignment>&, const aligned_allocator<T2, alignment>&)
        {
            return false;
        }

    };
}
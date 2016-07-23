/*
 * AlignedStlAllocator.h
 *
 *  Created on: Jun 14, 2013
 *      Author: dpajak
 */

#ifndef ALIGNEDSTLALLOCATOR_H_
#define ALIGNEDSTLALLOCATOR_H_

#include <memory>
#include "SystemCore.h"

namespace NVR {

template<typename T> class AlignedStlAllocator
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef T const * const_pointer;
    typedef T& reference;
    typedef T const & const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    pointer address(reference instance) const
    {
        return &instance;
    }

    const_pointer address(const_reference instance) const
    {
        return &instance;
    }

    size_type max_size(void) const
    {
        // The following has been carefully written to be independent of
        // the definition of size_t and to avoid signed/unsigned warnings.
        return (static_cast<size_type>(0) - static_cast<size_type>(1)) / sizeof(T);
    }

    pointer allocate(size_type n, void *hint = 0) const
    {
        (void)hint;
        return System::MemoryAlloc<T>(n, CACHELINE_ALIGNMENT);
    }

    void deallocate(pointer ptr, size_type n) const
    {
        (void)n;
        System::MemoryFree(ptr);
    }

    void construct(pointer ptr, const_reference instance) const
    {
        new (static_cast<void *>(ptr)) value_type(instance);
    }

    void destroy(const_pointer ptr) const
    {
        ptr->~value_type();
    }

    // The following must be the same for all allocators.
    template<typename U>
    struct rebind
    {
        typedef AlignedStlAllocator<U> other;
    };

};

}

#endif /* ALIGNEDSTLALLOCATOR_H_ */

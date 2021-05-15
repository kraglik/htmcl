/**
 * kma.cl
 * Main Kernel Memory Allocator OpenCL implementation
 * Copyright (C) 2013-2014 Roy Spliet, Delft University of Technology
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301
 * USA
 */

#ifndef KMA_H
#define KMA_H

#ifndef CLHEAP_H
#define CLHEAP_H

#ifdef __OPENCL_CL_H
#include <stdbool.h>

/* silence IDE */
#define __global
#define __kernel
#define __constant
#else

#define NULL (uintptr_t) 0
typedef volatile uintptr_t __global vg_uptr_t;
struct clheap;

void heap_init(void __global *);
void __global *malloc(__global struct clheap *heap, size_t);
void free(__global struct clheap *heap, uintptr_t);

#endif /* __OPENCL_CL_H */
#endif /* CLHEAP_H */


#ifndef CLINDEXEDQUEUE_H
#define CLINDEXEDQUEUE_H

#ifdef __OPENCL_CL_H
#include <stdbool.h>
#include <stdint.h>

/* silence IDE */
#define __global
#define __kernel
#define __constant
#else
#define uint32_t unsigned int
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#endif

typedef struct{
	char __global *base;
	size_t stride;
	volatile uint32_t head; /**< Head ptr of the queue */
	volatile uint32_t tail; /**< Tail ptr of the queue */
} clIndexedQueue;

typedef struct {
	/** Pointer to the next item in the queue. Must be 32-bit aligned, bit 0
	 * is used as "poison" bit when dequeuing.
	 * Under the condition that the "poison" bit 0 remains 1, the clqueue
	 * item can be part of a union. */
	volatile uint32_t next;
} clIndexedQueue_item;

#ifdef __OPENCL_CL_H
typedef volatile uintptr_t vg_uptr_t;
typedef struct  {
	uint32_t base;
	uint32_t stride;
	uint32_t head;
	uint32_t tail;
} clIndexedQueue_32;

typedef struct  {
	uint64_t base;
	uint64_t stride;
	uint32_t head;
	uint32_t tail;
} clIndexedQueue_64;

extern cl_mem clIndexedQueue_create(cl_device_id, cl_context, cl_command_queue,
		cl_program, cl_mem base, cl_uint stride_l2);
#else

#define NULL (uintptr_t) 0

extern void clIndexedQueue_init(void __global *, void __global *,
		uint32_t, void __global *);
extern int idxd_enqueue(clIndexedQueue __global *, clIndexedQueue_item __global *);
extern clIndexedQueue_item __global *idxd_dequeue(clIndexedQueue __global *);
#endif

#endif

#define KMA_SB_SIZE 4096	/**< Superblock size: 4KB */
#define KMA_SB_SIZE_LOG2 12	/**< Log 2 of superblock size: 2^12=4K */
#define KMA_SB_SIZE_BUCKETS KMA_SB_SIZE_LOG2 - 1 /**< Size buckets */

#ifdef __OPENCL_CL_H
typedef volatile uintptr_t vg_uptr_t;

char *heapBack;

struct kma_heap_32 {
	uint32_t bytes;
	clIndexedQueue_32 free;			/**< Free list */
	uint32_t sb[KMA_SB_SIZE_BUCKETS];	/**< SB hashtbl*/
};

struct kma_heap_64 {
	uint64_t bytes;
	clIndexedQueue_64 free;			/**< Free list */
	uint64_t sb[KMA_SB_SIZE_BUCKETS];	/**< SB hashtbl*/
};

extern cl_mem kma_create(cl_device_id dev, cl_context ctx, cl_command_queue cq,
		cl_program prg, unsigned int);
int clheap_execute(cl_device_id, cl_context, cl_command_queue,cl_program,
		size_t);
#else
#if CL_BITNESS == 64
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#endif
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

/* A superblock consists of the following:
 * Header:		    state + next-pointer
 * Data:		    available size (SB size - header - bitfield)
 * free/allocated bitfield: padded to end, one bit for each block inside
 * 			    superblock
 *
 * Once full, it should be detached from the superblock hashtable. When all
 * blocks are free'd, the block should be returned to the free list.
 */
struct kma_sb {
	clIndexedQueue_item q;		/**< Link to the next queue item
					  !!!: Keep me on top!
					  or build a container_of()*/
	volatile unsigned int state; 	/**< Slots, slots free */
	unsigned int size;		/**< Size of a block */
	unsigned int padding;
	char data[KMA_SB_SIZE - 12 - sizeof(clIndexedQueue_item)];	/**< Rest of the header */
};

/* This heap administration will take up 1 superblock */
struct clheap {
	size_t bytes;
	clIndexedQueue free;				 	 /**< Free list */
	volatile struct kma_sb __global *sb[KMA_SB_SIZE_BUCKETS]; /**< SB hashtbl*/
};
#endif
#endif

#define uptr uintptr_t
#define POISON 1

struct mem_item {
	clIndexedQueue_item q;
	uint32_t filler;
	uintptr_t thisobj;
};

/**
 * clIndexedQueue_ptr2idx() - Convert a pointer and tag to an index value
 * @q: Indexed queue object
 * @ptr: Pointer to convert
 * This helper function produces a uint32_t variable with the following bit-layout:
 * 0     Poison
 * 21:1  Index (base + stride * (n-1)
 * 32:22 Tag
 *
 * Returns 0 on failure.
 */
uint32_t
clIndexedQueue_ptr2idx(clIndexedQueue __global *q, void __global *ptr)
{
	uptr idx = (uptr) ptr;

	/* Before the base isn't possible */
	if(idx < (uptr) q->base)
		return 0;

	idx -= (uptr) q->base;
	/* Does it align to a stride boundary nicely? */
	if(idx & ((1 << q->stride) - 1))
		return 0;

	idx >>= q->stride;
	idx++;
	/* Does the index still fit? */
	if(idx > ((1 << 20) - 1))
		return 0;

	return (uint32_t) idx << 1;
}

/**
 * clIndexedQueue_idx2ptr() - Convert a pointer and tag to an index value
 * @q: Indexed queue object
 * @idx: Index to convert
 */
inline void __global*
clIndexedQueue_idx2ptr(clIndexedQueue __global *q, uint32_t idx)
{
	size_t i = idx;
	idx >>= 1;
	idx &= 0xfffff;
	if(idx == 0)
		return 0;

	idx--;

	i = idx;
	i <<= q->stride;

	return &q->base[i];
}

#define PTR(i,t) ((i & 0x1ffffe) | ((((t)+1) & 0x7ff) << 22))
#define TAG(i) ((i >> 22) & 0x7ff)
#define IDX(i) ((i & 0x1ffffe) >> 1)

__kernel void
clIndexedQueue_init(void __global *queue, void __global *base,
		uint32_t stride_l2, void __global *i)
{
	clIndexedQueue __global *q = (clIndexedQueue __global *) queue;
	uint32_t idx, tag;
	clIndexedQueue_item __global *item = (clIndexedQueue_item __global *)i;

	q->base = base;
	q->stride = stride_l2;

	tag = TAG(item->next);
	item->next = PTR(0, tag);
	idx = clIndexedQueue_ptr2idx(q, item);

	q->head = PTR(idx, tag);
	q->tail = PTR(idx, tag);
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

/**
 * idxd_enqueue() - Add an item to the indexed queue
 * @q: Queue to add the item to
 * @item: Item to add to the queue
 * @return 1 iff enqueuing succeeded, 0 otherwise
 */
int
idxd_enqueue(clIndexedQueue __global *q, clIndexedQueue_item __global *item)
{
	clIndexedQueue_item __global *tail;
	unsigned int ret = 0;
	uint32_t idx, tag,
	tailidx,
	nextidx;

	if(item == NULL)
		return 0;

	tag = TAG(item->next);
	item->next = PTR(0, tag-1);
	tag++;
	idx = clIndexedQueue_ptr2idx(q, item);

	while(1) {
		tailidx = q->tail;
		mem_fence(CLK_GLOBAL_MEM_FENCE);

		tail = (clIndexedQueue_item __global *)
				clIndexedQueue_idx2ptr(q, tailidx);
		nextidx = tail->next;
		mem_fence(CLK_GLOBAL_MEM_FENCE);

		/* Did I read a consistent state? */
		if(q->tail == tailidx) {
			if(IDX(nextidx) == 0) {
				tag = TAG(nextidx);
				if(atom_cmpxchg(&tail->next, nextidx, PTR(idx, tag)) == nextidx) {
					mem_fence(CLK_GLOBAL_MEM_FENCE);
					ret = 1;
					break;
				}
			} else {
				tag = TAG(tailidx);
				atom_cmpxchg(&q->tail, tailidx, PTR(nextidx, tag));
				mem_fence(CLK_GLOBAL_MEM_FENCE);
			}
		}
	}
	tag = TAG(tailidx);
	atom_cmpxchg(&q->tail, tailidx, PTR(idx, tag));
	mem_fence(CLK_GLOBAL_MEM_FENCE);

	return ret;
}

/**
 * idxd_dequeue() - Remove and return the next item in the queue
 * @q: Queue to get the item from
 * @return The next queue item, NULL on failure.
 */
clIndexedQueue_item __global *
idxd_dequeue(clIndexedQueue __global *q)
{
	clIndexedQueue_item __global *head;
	uint32_t 	tag,
			nextidx,
			tailidx,
			headidx;

	while(1) {
		headidx = q->head;
		mem_fence(CLK_GLOBAL_MEM_FENCE);
		head = (clIndexedQueue_item __global *)
				clIndexedQueue_idx2ptr(q, headidx);
		tailidx = q->tail;
		nextidx = head->next;
		mem_fence(CLK_GLOBAL_MEM_FENCE);

		if(headidx == q->head) {
			if(IDX(headidx) == IDX(tailidx)) {
				if(IDX(nextidx) == 0) {
					return NULL;
				}
				tag = TAG(tailidx);
				atom_cmpxchg(&q->tail, tailidx, PTR(nextidx, tag));
				mem_fence(CLK_GLOBAL_MEM_FENCE);
			} else {
				tag = TAG(headidx);
				if(atom_cmpxchg(&q->head, headidx, PTR(nextidx, tag)) == headidx)
					break;
			}
		}

	}
	mem_fence(CLK_GLOBAL_MEM_FENCE);
	return head;
}

/* Test enqueueing. */
__kernel void
clIndexedQueue_test_enqueue(void __global *queue, unsigned int __global *mem)
{
	clIndexedQueue __global *q = (clIndexedQueue __global *) queue;
	size_t pid = 0, j;
	unsigned int i;
	struct mem_item __global *item, *prev;

	/* First find global unique ID */
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}
	pid++;

	item = (struct mem_item __global *) &mem[pid * 4];

	idxd_enqueue(q, &item->q);

	if(pid == 1) {
		prev = (struct mem_item __global *) &mem[0];
		prev->thisobj = (uintptr_t) &mem[0];
	}
	item->thisobj = (uintptr_t) item;
}

/* Test enqueueing, dequeueing. */
__kernel void
clIndexedQueue_test_dequeue(void __global *queue, unsigned int __global *mem)
{
	clIndexedQueue __global *q = (clIndexedQueue __global *) queue;
	size_t pid = 0, j;
	unsigned int i;
	struct mem_item __global *item, *prev;

	/* First find global unique ID */
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}
	pid++;

	item = (struct mem_item __global *) &mem[pid * 4];
	item->thisobj = (uintptr_t) item;
	if(pid == 1) {
		prev = (struct mem_item __global *) &mem[0];
		prev->thisobj = (uintptr_t) &mem[0];
	}
	idxd_enqueue(q, (clIndexedQueue_item __global *) item);

	/* Do the shuffle */
	for(i = 0; i < 10; i++) {
		item = (struct mem_item __global *) idxd_dequeue(q);
		if(item != NULL) {
			if(!idxd_enqueue(q, (clIndexedQueue_item __global *) item)) {
			}
		} else {
		}
	}
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

/** Initialise the heap
 * @param heap The heap to initialise
 * Sets the pointers to null, initialises the free list with all pages
 */
__kernel void
clheap_init(void __global *hp, const unsigned long n_bytes)
{
	__global struct clheap *heap = (__global struct clheap *)hp;
	heap->bytes = n_bytes;
	struct kma_sb __global *sb;
	char __global *ptr;
	unsigned int pages;
	unsigned int i;

	/* Empty the superblock hashtable */
	for(i = 0; i < KMA_SB_SIZE_BUCKETS; i++) {
		heap->sb[i] = NULL;
	}

	/* Add all pages to the free list and initialise them */
	pages = (heap->bytes >> KMA_SB_SIZE_LOG2) - 1;
	ptr = (char __global *)heap;
	ptr += sizeof(struct clheap);
	sb = (struct clSuperBlock __global *)ptr;

	/* Initialise the free-list */
	clIndexedQueue_init(&heap->free, &sb[0], KMA_SB_SIZE_LOG2, &sb[0]);
	for(i = 1; i < pages; i++) {
		idxd_enqueue(&heap->free, &sb[i].q);
	}
}

__kernel void
clheap_init_step_1(void __global *hp, const unsigned long n_bytes)
{
	__global struct clheap *heap = (__global struct clheap *)hp;
	struct kma_sb __global *sb;
	char __global *ptr;
	unsigned int pages;
    heap->bytes = n_bytes;

    /* Empty the superblock hashtable */
    for(unsigned int i = 0; i < KMA_SB_SIZE_BUCKETS; i++) {
        heap->sb[i] = NULL;
    }

    ptr = (char __global *)heap;
    ptr += sizeof(struct clheap);
    sb = (struct clSuperBlock __global *)ptr;

    /* Initialise the free-list */
    clIndexedQueue_init(&heap->free, &sb[0], KMA_SB_SIZE_LOG2, &sb[0]);
}

__kernel void
clheap_init_step_2(
    void __global *hp,
    const unsigned long start_page,
    const unsigned long max_step
) {
	__global struct clheap *heap = (__global struct clheap *)hp;
	struct kma_sb __global *sb;
	char __global *ptr;
	unsigned int pages;

    /* Add all pages to the free list and initialise them */
    pages = (heap->bytes >> KMA_SB_SIZE_LOG2) - 1;
    ptr = (char __global *)heap;
    ptr += sizeof(struct clheap);
    sb = (struct clSuperBlock __global *)ptr;

	for(long i = start_page; i < pages; i++) {
	    if (i - start_page >= max_step) {
	        break;
	    }

		idxd_enqueue(&heap->free, &sb[i].q);
	}
}


/* For given sbid, return the size of a block in bytes */
size_t
_kma_size_by_sbid(int block)
{
	size_t size, total, bytes;
	unsigned int items, alloc_bytes;

	if(block > KMA_SB_SIZE_BUCKETS || block < 0)
		return 0;

	/* Blocks of 1 and 2 bytes don't exist for this malloc */
	block += 2;

	/* A bit of heuristic: For smaller blocks (sqrt(KMA_SB_SIZE))
	 * sacrifice one or two at the end of the superblock. */
	if(block <= (KMA_SB_SIZE_LOG2 >> 1)) {
		return 1 << (block);
	}

	/* For bigger blocks, adjust size to waste as little as possible
	 * 1. Calculate number of items when size is 2^(block+2)
	 * 2. Find total amount of space left when subtracting "alloc bits"
	 * 3. Divide this by the number of blocks*/
	bytes = 1 << block;
	items = KMA_SB_SIZE / bytes;

	/* Bytes rounded up to the nearest doubleword */
	alloc_bytes = items >> 3;
	alloc_bytes += (items & 0x7) ? 1 : 0;
	alloc_bytes += (alloc_bytes & 0x3) ? (4 - (alloc_bytes & 0x3)) : 0;

	/* Total space */
	total = KMA_SB_SIZE - (12 + alloc_bytes);
	size = total / items;

	return size & ~0x3;
}

/** For given size, return the superblock ID in the array
 * @param size size of the desired block
 * Returns the desired superblock id, -1 if no such superblock exists */
int
_kma_slots_by_size(size_t size)
{
	int slots;
	size_t space;

	space = KMA_SB_SIZE - 12;

	/* First approx */
	slots = space / size;
	/* Find out how many doublewords we need at the end of the block */
	if(slots & 0x1f)
		slots += 32;
	slots >>= 3;
	slots &= ~0x3;
	space -= slots;

	/* Recalculate with new space */
	return space / size;
}

/** For given size, return the superblock ID in the array
 * @param size size of the desired block
 * Returns the desired superblock id, -1 if no such superblock exists */
int
_kma_sbid_by_size(size_t size)
{
	int sbid, i;
	int test;
	size_t ssize;

	if(size > KMA_SB_SIZE - 16)
		return -1;

	ssize = size >> 2;

	/* First find a good approximation */
	for(i = 0; i < KMA_SB_SIZE_BUCKETS; i++, ssize >>= 1) {
		if(ssize & 1)
			sbid = i;
	}

	/* When block size increases, this no longer holds */
	while(true) {
		test = _kma_size_by_sbid(sbid);
		if(size > test && test > 0) {
			sbid++;
		} else if (size < _kma_size_by_sbid(sbid - 1)) {
			sbid--;
		} else {
			return sbid;
		}
	}
}

struct clSuperBlock __global *
_kma_reserve_block(__global struct clheap *heap, int block,
		unsigned int *slot)
{
	volatile struct kma_sb __global *cursor;
	unsigned int state, state_old, slots, i;
	volatile unsigned int __global *abits_ptr;

	if(block < 0 || block >= KMA_SB_SIZE_BUCKETS)
		return NULL;

	while(1) {
		/* Is there a superblock available */
		cursor = (struct kma_sb __global *)
				atom_cmpxchg((volatile uintptr_t __global *) &heap->sb[block], 0, POISON);
		mem_fence(CLK_GLOBAL_MEM_FENCE);
		if(cursor == 0) {
			/* No, let's reserve one */
			cursor = (struct kma_sb __global *)
					idxd_dequeue(&heap->free);
			if(!cursor) {
				/* No free pages left, return NULL */
				atom_xchg((volatile uintptr_t __global *) &heap->sb[block], 0);
				mem_fence(CLK_GLOBAL_MEM_FENCE);
				return NULL;
			}

			/* Reserve one for me */
			cursor->size = _kma_size_by_sbid(block);
			slots = _kma_slots_by_size(cursor->size);
			state = (slots << 16) | (slots - 1);
			cursor->state = state;

			/* Set all allocation bits to 0 (unallocated) */
			abits_ptr = (unsigned int __global *)cursor + (KMA_SB_SIZE >> 2);
			for(i = 0; i < slots; i += 32) {
				abits_ptr--;
				*abits_ptr = 0;
			}
			mem_fence(CLK_GLOBAL_MEM_FENCE);

			atom_cmpxchg((volatile uintptr_t __global *) &heap->sb[block], POISON, (uintptr_t) cursor);
			*slot = slots - 1;
			mem_fence(CLK_GLOBAL_MEM_FENCE);
			return cursor;
		}

		if((uintptr_t) cursor > POISON) {
			/* First reserve a slot */
			state_old = atom_add(&cursor->state, 0);
			mem_fence(CLK_GLOBAL_MEM_FENCE);
			state = state_old & 0xffff;
			slots = (state_old & 0xffff0000) >> 16;
			if(state == 0)
				continue;

			/* Decrease counter by 1 */
			state--;
			*slot = state;
			state |= (state_old & 0xffff0000);

			if(atom_cmpxchg(&cursor->state, state_old, state) != state_old)
				continue;
			mem_fence(CLK_GLOBAL_MEM_FENCE);

			/* If this was the last block in the SB, unlink */
			if((state & 0xffff) == 0) {
				atom_xchg((__global volatile uintptr_t *)&heap->sb[block], 0);
				mem_fence(CLK_GLOBAL_MEM_FENCE);
			}
			//heap->sb[8] += 1;
			return cursor;

		}
	}
}

/*
 * Return a pointer to a free block
 * @param heap Heap to allocate from
 * @pre Block has been reserved in state
 */
void __global *
_kma_get_block(struct kma_sb __global *sb, unsigned int slot)
{
	unsigned int abits = 0;
	volatile unsigned int __global *abits_ptr;
	unsigned int slots;
	uintptr_t ptr;
	unsigned int slot_orig = slot;

	slots = ((sb->state & 0xffff0000) >> 16);

	while(true) {
		abits_ptr = (volatile unsigned int __global *)sb;
		abits_ptr += (KMA_SB_SIZE >> 2);
		abits_ptr -= (slot >> 5);
		if(slot & 0x1f)
			abits_ptr--;
		if(slot)
			abits = *abits_ptr;
		abits >>= (slot & 0x1f);
		for(; slot < slots; slot++, abits >>= 1) {
			if((slot & 0x1f) == 0) {
				abits_ptr--;
				mem_fence(CLK_GLOBAL_MEM_FENCE);
				abits = *abits_ptr;
			}

			if((abits & 0x1) == 0) {
				/* Try setting the bit */
				if((atom_or(abits_ptr, (1 << (slot & 0x1f))) & (1 << (slot & 0x1f))) == 0) {
					mem_fence(CLK_GLOBAL_MEM_FENCE);
					/* Gotcha, I have block i */
					ptr = (uintptr_t) &sb->data;
					ptr += (slot * sb->size);
					*(unsigned int __global *)ptr = slot_orig;
					return (void __global *)ptr;
				}
			}
		}
		slot = 0;
	}
}

/** Allocate memory
 * @param heap Heap
 * @param size Size of the desired block
 */
void __global *
malloc(__global struct clheap *heap, size_t size)
{
	int block;
	unsigned int slot,i;
	struct kma_sb __global *sb;
	/* Sizes all come in log2 for now. This means a lot of wastage for
	 * medium-sized memory blocks.
	 * Earlier experiments showed that traversing a linked list could lead
	 * to a corrupted cursor, with unpredictable behaviour. We can improve
	 * by increasing the granularity and adding more size buckets, at the
	 * cost of possibly more internal fragmentation
	 *
	 * Let's find a suitable superblock */
	block = _kma_sbid_by_size(size);
	if(block < 0)
		return NULL;

	sb = _kma_reserve_block(heap, block, &slot);
	if(!sb) {
		return NULL;
	}

	return _kma_get_block(sb, slot);
}

void
free(__global struct clheap *heap, uintptr_t block)
{
	unsigned int size, mask;
	volatile struct kma_sb __global *sb;
	uintptr_t first_sb, off;
	unsigned int state_old, state, slots, sbid;
	bool enq;
	volatile unsigned int __global *abits_ptr;
	volatile unsigned int __global *b = (volatile unsigned int __global *) block;

	if(block == NULL)
		return;

	/* Find superblock */
	first_sb = ((uintptr_t) heap + sizeof(struct clheap));
	off = block - first_sb;
	mask = (1 << KMA_SB_SIZE_LOG2) - 1;

	/* Find size of block */
	sb = (volatile struct clSuperBlock __global *)(first_sb + (off & ~mask));
	size = sb->size;

	/* Index of this block */
	block -= (uintptr_t) sb;
	block -= (12 + sizeof(clIndexedQueue_item));
	block /= size;

	/* Update the "taken" bit
	 * XXX: If you try to free a block that isn't taken, "free slots"
	 * does get incremented. Corrupting the state! */
	abits_ptr = (volatile unsigned int __global *)(sb+1);
	abits_ptr -= ((block >> 5) + 1);
	*b = atom_and(abits_ptr, ~(1 << (block & 0x1f)));

	/* Update free slots */
	do {
		mem_fence(CLK_GLOBAL_MEM_FENCE);
		state_old = atom_add(&sb->state, 0);

		state = state_old & 0xffff;
		slots = (state_old & 0xffff0000) >> 16;
		state++;

		/* Enqueue this superblock and "unlink" */
		if(state == slots) {
			enq = 1;
			state = 0;
		} else {
			enq = 0;
		}

		state |= (slots << 16);
		mem_fence(CLK_GLOBAL_MEM_FENCE);
	} while (atom_cmpxchg(&sb->state, state_old, state) != state_old);
	mem_fence(CLK_GLOBAL_MEM_FENCE);

	/* find the right sbid and enqueue superblock if required */
	if(enq) {
		sbid = _kma_sbid_by_size(sb->size);
		atom_cmpxchg((volatile uintptr_t __global *)&heap->sb[sbid], (uintptr_t) sb, 0);
		mem_fence(CLK_GLOBAL_MEM_FENCE);
		idxd_enqueue(&heap->free, &sb->q);
	} else {
		/* Try to re-attach to avoid wasting too much mem */
		//atom_cmpxchg((volatile uintptr_t __global *)&heap->sb[sbid], NULL, (uintptr_t) sb);
		//mem_fence(CLK_GLOBAL_MEM_FENCE);
	}
}

/******************************
 * Tests
 *****************************/

/* Create lookup table for sbid->size */
__kernel void
kma_test_size_by_sbid(unsigned int __global *array)
{
	unsigned int pid, i, j;

	/* First find global unique ID */
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}

	array[pid] = _kma_size_by_sbid(pid);
}

/* Create lookup table for sbid->size
__kernel void
clSBMalloc_test_slots_by_sbid(unsigned int __global *array)
{
	unsigned int pid, i, j;

	// First find global unique ID
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}

	array[pid] = _clSBMalloc_slots_by_sbid(pid);
} */

/* Create lookup table for sbid->size */
__kernel void
kma_test_sbid_by_size(unsigned int __global *array)
{
	unsigned int pid, i, j;

	/* First find global unique ID */
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}

	array[pid] = _kma_sbid_by_size(pid);
}

__kernel void
kma_test_malloc(struct clheap __global *heap, unsigned int iters)
{
	size_t pid = 0, j;
	unsigned int i;
	volatile size_t __global *block;

	/* First find global unique ID */
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}

	for(i = 0; i < iters; i++) {
		block = (size_t __global *)malloc(heap, sizeof(size_t));
		if(!block) {
			return;
		}
		block[0] = pid;
		//if(pid == 0)
			free(heap, (uintptr_t) block);
	}
	/*barrier(CLK_GLOBAL_MEM_FENCE);
	heap->sb[7] = heap->sb[8];
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(pid == 1) {
		block2 = malloc(heap, sizeof(size_t));
		mem_fence(CLK_GLOBAL_MEM_FENCE);
		if(pid == 18)
			block2[0] = pid;
		//free(heap, (uintptr_t) block);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(pid == 18)
		block[0] = pid;*/
}

__kernel void
kma_test_malloc_lowvar(struct clheap __global *heap, unsigned int iters)
{
	size_t pid = 0, j;
	unsigned int i;
	unsigned int __global *block, *block2;

	/* First find global unique ID */
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}

	for(i = 0; i < iters; i++) {
		block = (unsigned int __global *)malloc(heap,(pid & 0x1) ? 8 : 16);
		if(!block) {
			return;
		}
		block[0] = pid;
		if(pid == 0) {
			block2 = (unsigned int __global *)malloc(heap, 4000);
			if(!block2) {
				return;
			}
			block2[8] = 4919;
			free(heap, (uintptr_t) block2);
		}
		free(heap, (uintptr_t) block);
	}
}

__kernel void
kma_test_malloc_highvar(struct clheap __global *heap, unsigned int iters)
{
	size_t pid = 0, j;
	unsigned int i, amount;
	unsigned int __global *block, *block2;

	/* First find global unique ID */
	for(i = 0, j = 1; i < get_work_dim(); i++) {
		pid += j * get_global_id(i);
		j *= get_global_size(i);
	}

	for(i = 0; i < iters; i++) {
		amount = pid + i;
		amount = (amount % 5);
		amount = 4 << amount;
		block = (unsigned int __global *)malloc(heap, amount);
		if(!block) {
			return;
		}
		block[0] = 4919;
		if(pid == 0) {
			block2 = (unsigned int __global *)malloc(heap, 4000);
			if(!block2) {
				return;
			}
			block2[8] = 4919;
			free(heap, (uintptr_t) block2);
		}
		free(heap, (uintptr_t) block);
	}
}

/* Only for OpenCL 1.2+ */
//#if !(CL_PLATFORM==2)
//__kernel void
//clSBMalloc_test_heap_sbs(struct clheap __global *heap)
//{
//	unsigned int pages, i;
//	char __global *hp;
//	struct clSuperBlock __global *sb;
//
//	hp = ((char __global *)heap) + sizeof(struct clheap);
//	sb = (struct clSuperBlock __global *) hp;
//
//	pages = (heap->bytes >> CLSBM_SB_SIZE_LOG2) - 1;
//
//	for(i = 0; i < pages; i++) {
//		if(sb->q.next == 0 && heap->free.tail != sb)
//			//printf("Orphaned superblock: %08x\n", sb);
//		sb++;
//	}
//}
//#endif



kernel void
get_clheap_size(global unsigned int* out) {
    out[0] = sizeof(struct clheap);
}

typedef struct vec2 {
    unsigned int x, y;
    global volatile unsigned int* xs;
} vec2;


kernel void
test_allocations(global struct clheap* heap, global int* results) {

    unsigned int id = get_global_id(0);

    for (unsigned int i = 0; i < 128; i++) {

        global vec2* vec = (global vec2*) malloc(heap, sizeof(vec2));


        vec->x = id;
        vec->y = id + id;
        vec->xs = (global unsigned int*)malloc(heap, sizeof(unsigned int));

        results[id] = vec->x - vec->y;

        free(heap, vec->xs);
        free(heap, vec);

    }
}

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Memory-aware LRU Cache function decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A modification of the builtin ``functools.lru_cache`` decorator that takes an
additional keyword argument, ``use_memory_up_to``. The cache is considered full
if there are fewer than ``use_memory_up_to`` bytes of memory available.

If ``use_memory_up_to`` is set, then ``maxsize`` has no effect.

Uses the ``vmem`` module to get the available memory.
"""

import vmem
try:
    from functools import RLock, namedtuple
except ImportError:
    from threading import RLock
    from collections import namedtuple
from functools import update_wrapper

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.

    """

    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue

def _make_key(args, kwds, typed,
             kwd_mark = (object(),),
             fasttypes = {int, str, frozenset, type(None)},
             sorted=sorted, tuple=tuple, type=type, len=len):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)

def lru_cache(maxsize=128, typed=False, use_memory_up_to=False):
    """Least-recently-used cache decorator.

    *use_memory_up_to* is an integer representing the number of bytes of memory
    that must be available on the system in order for a value to be cached. If
    it is set, *maxsize* has no effect.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    """
    if use_memory_up_to:
        maxsize=None

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    # Constants shared by all lru cache instances:
    sentinel = object()          # unique object used to signal cache misses
    make_key = _make_key         # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3   # names for the link fields

    def decorating_function(user_function):
        cache = {}

        class nonloc:
            hits = 0
            misses = 0
            root = [] # root of the circular doubly linked list
            full = False

        cache_get = cache.get    # bound method to lookup a key or return None
        lock = RLock()           # because linkedlist updates aren't threadsafe
        nonloc.root[:] = [nonloc.root, nonloc.root, None, None]     # initialize by pointing to self

        if use_memory_up_to:

            def wrapper(*args, **kwds):
                # Size limited caching that tracks accesses by recency
                key = make_key(args, kwds, typed)
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # Move the link to the front of the circular queue
                        link_prev, link_next, _key, result = link
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = nonloc.root[PREV]
                        last[NEXT] = nonloc.root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = nonloc.root
                        nonloc.hits += 1
                        return result
                result = user_function(*args, **kwds)
                with lock:
                    if key in cache:
                        # Getting here means that this same key was added to the
                        # cache while the lock was released.  Since the link
                        # update is already done, we need only return the
                        # computed result and update the count of misses.
                        pass
                    elif nonloc.full:
                        # Use the old root to store the new key and result.
                        oldroot = nonloc.root
                        oldroot[KEY] = key
                        oldroot[RESULT] = result
                        # Empty the oldest link and make it the new root.
                        # Keep a reference to the old key and old result to
                        # prevent their ref counts from going to zero during the
                        # update. That will prevent potentially arbitrary object
                        # clean-up code (i.e. __del__) from running while we're
                        # still adjusting the links.
                        nonloc.root = oldroot[NEXT]
                        oldkey = nonloc.root[KEY]
                        oldresult = nonloc.root[RESULT]
                        nonloc.root[KEY] = nonloc.root[RESULT] = None
                        # Now update the cache dictionary.
                        del cache[oldkey]
                        # Save the potentially reentrant cache[key] assignment
                        # for last, after the root and links have been put in
                        # a consistent state.
                        cache[key] = oldroot
                    else:
                        # Put result in a new link at the front of the queue.
                        last = nonloc.root[PREV]
                        link = [last, nonloc.root, key, result]
                        last[NEXT] = nonloc.root[PREV] = cache[key] = link
                        nonloc.full = (vmem.virtual_memory().available
                                < use_memory_up_to)
                    nonloc.misses += 1
                return result

        elif maxsize == 0:

            def wrapper(*args, **kwds):
                # No caching -- just a statistics update after a successful call
                result = user_function(*args, **kwds)
                nonloc.misses += 1
                return result

        elif maxsize is None:

            def wrapper(*args, **kwds):
                # Simple caching without ordering or size limit
                key = make_key(args, kwds, typed)
                result = cache_get(key, sentinel)
                if result is not sentinel:
                    nonloc.hits += 1
                    return result
                result = user_function(*args, **kwds)
                cache[key] = result
                nonloc.misses += 1
                return result

        else:

            def wrapper(*args, **kwds):
                # Size limited caching that tracks accesses by recency
                key = make_key(args, kwds, typed)
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # Move the link to the front of the circular queue
                        link_prev, link_next, _key, result = link
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = nonloc.root[PREV]
                        last[NEXT] = nonloc.root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = nonloc.root
                        nonloc.hits += 1
                        return result
                result = user_function(*args, **kwds)
                with lock:
                    if key in cache:
                        # Getting here means that this same key was added to the
                        # cache while the lock was released.  Since the link
                        # update is already done, we need only return the
                        # computed result and update the count of misses.
                        pass
                    elif nonloc.full:
                        # Use the old root to store the new key and result.
                        oldroot = nonloc.root
                        oldroot[KEY] = key
                        oldroot[RESULT] = result
                        # Empty the oldest link and make it the new root.
                        # Keep a reference to the old key and old result to
                        # prevent their ref counts from going to zero during the
                        # update. That will prevent potentially arbitrary object
                        # clean-up code (i.e. __del__) from running while we're
                        # still adjusting the links.
                        nonloc.root = oldroot[NEXT]
                        oldkey = nonloc.root[KEY]
                        oldresult = nonloc.root[RESULT]
                        nonloc.root[KEY] = nonloc.root[RESULT] = None
                        # Now update the cache dictionary.
                        del cache[oldkey]
                        # Save the potentially reentrant cache[key] assignment
                        # for last, after the root and links have been put in
                        # a consistent state.
                        cache[key] = oldroot
                    else:
                        # Put result in a new link at the front of the queue.
                        last = nonloc.root[PREV]
                        link = [last, nonloc.root, key, result]
                        last[NEXT] = nonloc.root[PREV] = cache[key] = link
                        nonloc.full = (len(cache) >= maxsize)
                    nonloc.misses += 1
                return result

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CacheInfo(nonloc.hits, misses, maxsize, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            with lock:
                cache.clear()
                nonloc.root[:] = [nonloc.root, nonloc.root, None, None]
                nonloc.hits = nonloc.misses = 0
                nonloc.full = False

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)

    return decorating_function
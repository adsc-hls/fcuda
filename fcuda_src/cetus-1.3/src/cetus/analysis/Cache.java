package cetus.analysis;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Data structure implementing FIFO cache. Time-consuming manipulations can be
 * cached to save computation time at the price of additional memory usage.
 */
public class Cache<K,E> extends LinkedHashMap<K,E> {

    private static final long serialVersionUID = 16L;

    private int max_entries = 64;

    public Cache() {
        super();
    }

    public Cache(int max_entries) {
        super();
        this.max_entries = max_entries;
    }

    protected boolean removeEldestEntry(Map.Entry eldest) {
        return size() > max_entries;
    }

}

package cetus.analysis;

import java.util.*;
import cetus.hir.PrintTools;

/**
 * Class DFANode represents a node object to be used as workspace in any
 * data flow analysis. The "data" map can hold any generic type of contents
 * by mapping string-type keys to the contents. The "preds" map and the "succs"
 * map hold edge-specific data coupled with each predecessors or successors.
 * The keys of these two maps are useful as they are equivalent to the set
 * of predecessors and the set of successors for the node.
 */
public class DFANode {

    // Container for satellite objects
    private Map<String, Object> data;

    // Container for incoming edges
    private Map<DFANode, Object> preds;

    // Container for outgoing edges
    private Map<DFANode, Object> succs;

    /**
     * Constructs an empty node.
     */
    public DFANode() {
        data = new HashMap<String, Object>(1);
        preds = new LinkedHashMap<DFANode, Object>(1);
        succs = new LinkedHashMap<DFANode, Object>(1);
    }

    /**
     * Constructs a node with the specified key/data pair.
     *
     * @param key the key string.
     * @param data the associated data.
     */
    public DFANode(String key, Object data) {
        this();
        this.data.put(key,data);
    }

    /**
     * Returns the set of keys in the satellite data map.
     *
     * @return the set of keys.
     */
    public Set<String> getKeys() {
        return data.keySet();
    }

    /**
     * Returns the data in the node mapped by the key. The warnings are
     * suppressed since the java compiler, most of the time, generates
     * "unchecked cast" warnings at calls to this method. Hence, it is soley
     * pass writers' responsibility to use putData and getData consistently.
     *
     * @param key a string key.
     * @return the object mapped by the key. null if the key does not exist.
     */
    @SuppressWarnings("unchecked")
    public <T> T getData(String key) {
        return (T)data.get(key);
    }

    /**
     * Returns the first data while searching for the specified list of keys.
     *
     * @param keys the list of keys.
     * @return the data found first. null if the keys do not exist.
     */
    @SuppressWarnings("unchecked")
    public <T> T getData(List<String> keys) {
        Object ret = null;
        for (String key : keys) {
            if ((ret = data.get(key)) != null) {
                break;
            }
        }
        return (T)ret;
    }

    /**
     * Associates the given data with the specified key.
     *
     * @param key the key string.
     * @param data the associated data.
     */
    public void putData(String key, Object data) {
        this.data.put(key, data);
    }

    /**
     * Removes the data mapped by the specified key.
     *
     * @param key the key string.
     */
    public void removeData(String key) {
        this.data.remove(key);
    }

    /**
     * Returns the set of successor nodes.
     *
     * @return the set of successors.
     */
    public Set<DFANode> getSuccs() {
        return succs.keySet();
    }

    /**
     * Returns the set of predecessor nodes.
     *
     * @return the set of predecessors.
     */
    public Set<DFANode> getPreds() {
        return preds.keySet();
    }

    /**
     * Returns the successor data associated with the specified successor key.
     *
     * @param key the successor whose associated data is asked for.
     * @return the associated data for the key.
     */
    @SuppressWarnings("unchecked")
    public <T> T getSuccData(DFANode key) {
        return (T)succs.get(key);
    }

    /**
     * Returns the predecessor data associated with the specified predecessor
     * key.
     *
     * @param key the predecessor whose associated data is asked for.
     * @return the associated data for the key.
     */
    @SuppressWarnings("unchecked")
    public <T> T getPredData(DFANode key) {
        return (T)preds.get(key);
    }

    /**
     * Adds data associated with the outgoing edge from this node to the
     * successor node.
     *
     * @param succ the successor node.
     * @param value the associated data.
     */ 
    public void putSuccData(DFANode succ, Object value) {
        succs.put(succ, value);
    }

    /**
     * Adds data associated with the incoming edge from the predecessor node to
     * this node.
     *
     * @param pred the predecessor node.
     * @param value the associated data.
     */ 
    public void putPredData(DFANode pred, Object value) {
        preds.put(pred, value);
    }

    /**
     * Adds a predecessor node.
     *
     * @param pred the predecessor node.
     */ 
    public void addPred(DFANode pred) {
        preds.put(pred, null);
    }

    /**
     * Adds a successor node.
     *
     * @param succ the successor node.
     */ 
    public void addSucc(DFANode succ) {
        succs.put(succ, null);
    }

    /**
     * Removes a predecessor node.
     *
     * @param pred the predecessor node.
     */ 
    public void removePred(DFANode pred) {
        preds.remove(pred);
    }

    /**
     * Removes a successor node.
     *
     * @param succ the successor node.
     */ 
    public void removeSucc(DFANode succ) {
        succs.remove(succ);
    }

    /**
     * Returns a string for the graph.
     *
     * @return the listing of the contents of the graph in a string.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        String sep = PrintTools.line_sep;
        sb.append(this.hashCode());
        sb.append(":").append(sep).append("{").append(sep);
        for (String key : data.keySet()) {
            Object value = data.get(key);
            sb.append(" ").append(key).append(": ");
            if (value instanceof DFANode) {
                sb.append(value.hashCode()).append(sep);
            } else {
                sb.append(value).append(sep);
            }
        }
        sb.append("  preds={").append(sep);
        for (DFANode key : preds.keySet()) {
            sb.append("    ").append(key.hashCode()).append(": ");
            sb.append(preds.get(key)).append(sep);
        }
        sb.append("  }").append(sep).append("  succs={").append(sep);
        for (DFANode key: succs.keySet()) {
            sb.append("    ").append(key.hashCode()).append(": ");
            sb.append(succs.get(key)).append(sep);
        }
        sb.append("  }").append(sep);
        return sb.toString(); 
    }

    /**
     * Returns the string in dot format that represents the node.
     *
     * @param keys the comma-separated list of keys whose mapped data are
     *      printed.
     * @param num the number of total keys being searched for.
     * @return the string in dot format.
     */
    public String toDot(String keys, int num) {
        String label = "";
        String[] labels = keys.split(",");
        for (int i = 0, j = 0; i < labels.length; ++i) {
            Object found = data.get(labels[i]);
            if (found != null) {
                label += found + "\\n";
                if (++j >= num) {
                    label = label.replace("\"", "\\\"");
                    break;
                }
            }
        }
        return "[label=\"" + label + "\"]";
    }
}

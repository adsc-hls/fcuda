package cetus.hir;

import java.util.List;
import java.util.ArrayList;
import java.util.NoSuchElementException;

/**
* Alternative implementation to {@link DepthFirstIterator}, which supports
* depth-first iteration over a specific type when requested. Unlike {@code
* DepthFirstIterator}, {@code DFIterator} does not allocate internal storage
* to place next items to be searched for; it performs memory-less search by
* visiting the node in the IR tree to find the next item to be returned.
*/
public class DFIterator<E extends Traversable> {

    /** Initial size of the list of pruned types */
    private static final int DEFAULT_PRUNED_ON_SIZE = 4;

    /** List of types whose child nodes are skipped during iteration. */
    private List<Class<? extends Traversable>> pruned_on;

    /** The initial IR node of the iterator. */
    private Traversable root;

    /** The next IR node to be returned. */
    private Traversable next;

    /** The IR node type to be returned during iteration. */
    private Class<? extends Traversable> type;

    /**
    * Constructs a new iterator that returns any traversable nodes during
    * iteration.
    * @param root the initial node for the iteration.
    */
    public DFIterator(Traversable root) {
        this.root = root;
        pruned_on = null;
        type = Traversable.class;
        reset();
    }

    /**
    * Constructs a new iterator that returns the specified IR type during
    * iteration.
    * @param root the initial node for the iteration.
    * @param c the IR class type to be iterated over.
    */
    public DFIterator(Traversable root, Class<? extends Traversable> c) {
        this.root = root;
        pruned_on = null;
        type = c;
        reset();
    }

    /** 
    * Checks if there is a next element of the requested type.
    * @return true if there exist a next element of the requested type.
    */
    public boolean hasNext() {
        return next != null;
    }

    /**
    * Returns the next IR node.
    * @return the next IR node.
    * @exception NoSuchElementException no more elements found.
    */
    @SuppressWarnings("unchecked")
    public E next() {
        E ret = (E)next;
        try {
            next = findNext(ret);
        }
        catch(NullPointerException ex) {
            throw new NoSuchElementException();
        }
        return ret;
    }

    /**
    * Considers the specified IR class type as one whose child nodes are not
    * visited during iteration. This method is useful for reducing the cost
    * of iteration.
    * @param c the IR node type to be pruned on.
    */
    public void pruneOn(Class<? extends Traversable> c) {
        if (pruned_on == null) {
            pruned_on = new ArrayList<Class<? extends Traversable>>(
                    DEFAULT_PRUNED_ON_SIZE);
        }
        pruned_on.add(c);
    }

    /**
    * Initializes the iterator by placing the first item to be returned for a
    * call to {@link #next()}.
    */
    public void reset() {
        if (type.isInstance(root)) {
            next = root;
        } else {
            next = findNext(root);
        }
    }

    /**
    * Drives memory-less depth-first search by 1) searching the next item in the
    * subtree rooted at the current node {@code t} and 2) searching the
    * supertree disregarding the subtree mentioned in 1).
    * @param t the IR node where the search starts.
    * @return the next element of the requested type {@code E} or null if one
    * is not found.
    */
    @SuppressWarnings("unchecked")
    private E findNext(Traversable t) {
        E ret = findNext(t, 0);
        // Continue searching after pruning the subtree rooted at "t".
        if (ret == null) {
            Traversable child = t;
            Traversable parent = child.getParent();
            // NOTE: parent != root.getParent() implies parent != null
            while (ret == null && parent != root.getParent()) {
                int t_pos = Tools.identityIndexOf(parent.getChildren(), child);
                ret = findNext(parent, t_pos + 1);
                child = parent;
                parent = child.getParent();
            }
        }
        return ret;
    }

    /**
    * Performs depth-first search within the tree rooted at the current node
    * {@code t}, skipping the subtree rooted at {@code 0, 1, ..., (pos-1)}-th
    * child of the current node.
    * @param t the IR node where the search starts.
    * @param pos the position that masks subtrees that are already visited.
    * @return the next element of the requested type {@code E} or null if one
    * is not found.
    */
    @SuppressWarnings("unchecked")
    private E findNext(Traversable t, int pos) {
        E ret = null;
        List<Traversable> children = t.getChildren();
        if (!isPruned(t) && children != null) {
            for (int i = pos; i < children.size() && ret == null; i++) {
                Traversable child = children.get(i);
                if (child == null) {
                    continue;
                }
                if (type.isInstance(child)) {
                    ret = (E)child;
                } else {
                    ret = findNext(child, 0);
                }
            }
        }
        return ret;
    }

    /**
    * Tests if the specified traversable object belongs in a list of IR types
    * that are pruned on.
    * @param t the IR node to be tested.
    * @return true if it is.
    */
    private boolean isPruned(Traversable t) {
        if (pruned_on == null) {
            return false;
        }
        for (int i = 0; i < pruned_on.size(); i++) {
            if (pruned_on.get(i).isInstance(t)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Returns a list of traversed elements of type {@code E} using the current
    * iterator.
    * @return the collected list.
    */
    public List<E> getList() {
        List<E> ret = new ArrayList<E>();
        reset();
        while (hasNext()) {
            ret.add(next());
        }
        return ret;
    }

    /**
    * Forces the specified traversable element {@code next} to be returned when
    * {@link #next()} is called. Appropriate use of this method enables IR
    * traversal with modification in a well-defined manner. (experimental)
    * @param next the next traversable element to be returned.
    */
    public void setNext(E next) {
        if (IRTools.isAncestorOf(root, next)) {
            this.next = next;
        } else {
            throw new IllegalArgumentException();
        }
    }

}

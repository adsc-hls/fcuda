package cetus.hir;

import java.util.*;

/**
* Iterates over Traversable objects in breadth-first order.
*/
public class BreadthFirstIterator<E extends Traversable>
                                    extends IRIterator<E> {

    private Vector<Traversable> queue;

    private List<Class<? extends Traversable>> pruned_on;

    /**
    * Creates a new iterator with the specified initial object and the
    * optional list of pruned types.
    *
    * @param init The first object to visit.
    */
    public BreadthFirstIterator(Traversable init) {
        super(init);
        queue = new Vector<Traversable>();
        queue.add(init);
        pruned_on = null;
    }
    
    public boolean hasNext() {
        return !queue.isEmpty();
    }

    @SuppressWarnings("unchecked")
    public E next() {
        Traversable t = null;
        try {
            t = queue.remove(0);
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new NoSuchElementException();
        }
        if (t.getChildren() != null && !isPruned(t)) {
            List<Traversable> children = t.getChildren();
            int children_size = children.size();
            for (int i = 0; i < children_size; i++) {
                if (children.get(i) != null) {
                    queue.add(children.get(i));
                }
            }
        }
        return (E)t;
    }

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
    * Disables traversal from an object having the specified type. For example,
    * if traversal reaches an object with type <b>c</b>, it does not visit the
    * children of the object.
    *
    * @param c the object type to be pruned on.
    */
    public void pruneOn(Class<? extends Traversable> c) {
        if (pruned_on == null) {
            pruned_on = new ArrayList<Class<? extends Traversable>>(4);
        }
        pruned_on.add(c);
    }

    /**
    * Returns a linked list of objects of Class c in the IR
    *
    * @param c the object type to be collected.
    * @return the collected list.
    */
    @SuppressWarnings("unchecked")
    public LinkedList getList(Class<? extends Traversable> c) {
        LinkedList list = new LinkedList();
        while (hasNext()) {
            Object obj = next();
            if (c.isInstance(obj)) {
                list.add(obj);
            }
        }
        return list;
    }

    /**
    * Returns a set of objects of Class c in the IR
    *
    * @param c the object type to be collected.
    * @return the collected set.
    */
    @SuppressWarnings("unchecked")
    public Set getSet(Class<? extends Traversable> c) {
        HashSet set = new HashSet();
        while (hasNext()) {
            Object obj = next();
            if (c.isInstance(obj)) {
                set.add(obj);
            }
        }
        return set;
    }

    /**
    * Resets the iterator by setting the current position to the root object.
    * The pruned types are not cleared.
    */
    public void reset() {
        queue.clear();
        queue.add(root);
    }

    /**
    * Unlike the <b>reset</b> method, <b>clear</b> method also clears the
    * pruned types.
    */
    public void clear() {
        queue.clear();
        pruned_on.clear();
        queue.add(root);
    }
}

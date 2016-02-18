package cetus.hir;

import java.util.List;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Arrays;

/**
* Peforms a post-order traversal over a Traversable object. This type
* of traversal usually requires larger internal storage space than
* <b>DepthFirstIterator</b> or <b>BreadthFirstIterator</b> does since it
* needs to store all the visited objects while reaching the first element
* to be visited which is usually a leaf node.
*/
public class PostOrderIterator<E extends Traversable> extends IRIterator<E> {

    private LinkedList<Traversable> queue;

    private List<Class<? extends Traversable>> pruned_on;

    /**
    * Creates a new iterator with the specified root object.
    *
    * @param root The root object for the traversal.
    */
    public PostOrderIterator(Traversable root) {
        super(root);
        queue = new LinkedList<Traversable>();
        populate(root);
        pruned_on = null;
    }

    /**
    * Creates a new iterator with the specified root object and the array of
    * classes that needs to be pruned on.
    *
    * @param root The root object for the traversal.
    * @param pruned_on The list of object types to be pruned on.
    */
    public PostOrderIterator(Traversable root,
                             Class<? extends Traversable>[] pruned_on) {
        super(root);
        queue = new LinkedList<Traversable>();
        this.pruned_on = Arrays.asList(pruned_on);
        populate(root);
    }
    
    public boolean hasNext() {
        return !queue.isEmpty();
    }

    @SuppressWarnings("unchecked")
    public E next() {
        Traversable t = null;
        // will throw NoSuchElementException on failure.
        return (E)queue.remove();
    }

    private void populate(Traversable t) {
        if (t.getChildren() != null && !isPruned(t)) {
            List<Traversable> t_children = t.getChildren();
            int t_children_size = t_children.size();
            for (int i = 0; i < t_children_size; i++) {
                if (t_children.get(i) != null) {
                    populate(t_children.get(i));
                }
            }
        }
        queue.add(t);
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
    * Resets the iterator by setting the current position to the root object.
    * The pruned types are not cleared.
    */
    public void reset() {
        queue.clear();
        populate(root);
    }

    /**
    * Unlike the <b>reset</b> method, <b>clear</b> method also clears the
    * pruned types.
    */
    public void clear() {
        queue.clear();
        pruned_on.clear();
        populate(root);
    }

}

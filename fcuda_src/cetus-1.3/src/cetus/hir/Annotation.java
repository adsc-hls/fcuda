package cetus.hir;

import java.util.*;

/**
* Annotation is the base class of any annotation type used in Cetus.
* Unlike the implementation in the past versions of Cetus, this class is
* separated from the IR tree to clarify usage of annotations along with
* their associated IR (either Declaration or Statement).
*/
public abstract class Annotation extends HashMap<String, Object> {

    // Possible printed position relative to the associated IR:
    // BEFORE: before the IR
    // WITH  : after the IR without line breaking
    // AFTER : after the IR
    protected static final int BEFORE = -1;
    protected static final int WITH = 0;
    protected static final int AFTER = 1;

    // The relative position from the IR
    protected int position;

    // The annotatable object having this annotation.
    protected Annotatable ir;

    // Used when turning off printing. 
    protected boolean skip_print;

    /**
    * Constructs a new annotation.
    */
    protected Annotation() {
        super();
        position = BEFORE;
        ir = null;
        skip_print = false;
    }

    /**
    * Returns the annotated value with the specified key.
    * @param key the given string key.
    * @return the annotated value or null (if not present).
    */
    @SuppressWarnings("unchecked")
    public <T> T get(String key) {
        return (T)super.get(key);
    }

    /**
    * Sets the relative position from the associated IR with the given position.
    * @param position the new position.
    */
    public void setPosition(int position) {
        this.position = position;
    }

    /**
    * Returns a string representation of the annotation. All child classes
    * of Annotation should implement their own toString() method.
    * @return the string representation.
    */
    public abstract String toString();

    /**
    * Returns the string representation of this annotation if the given
    * position is equal to the position of this annotation.
    */
    public String toString(int position) {
        if (this.position == position) {
            return toString();
        } else {
            return "";
        }
    }

    /**
    * Returns a clone of this annotation object.
    * @return a cloned annotation.
    */
    @SuppressWarnings("unchecked")
    @Override
    public Annotation clone() {
        Annotation o = (Annotation)super.clone();      // super is cloneable.
        // Overwrite shallow copies.
        o.clear();
        Iterator<String> iter = keySet().iterator();
        while (iter.hasNext()) {
            String key = iter.next();
            Object val = get(key);
            o.put(key, cloneObject(val));
        }
        o.position = this.position;
        o.skip_print = this.skip_print;
        // ir are overwritten only by annotatable.annotate().
        return o;
    }

    /**
    * returns the deep copy of the given map
    */
    @SuppressWarnings("unchecked")
    private HashMap cloneMap(Map map) {
        Iterator<String> iter = map.keySet().iterator();
        HashMap clone = new HashMap();
        while (iter.hasNext()) {
            String key = iter.next();
            Object val = map.get(key);
            clone.put(key, cloneObject(val));
        }
        return clone;
    }

    /**
    * returns the deep copy of the given object (which could be String,
    * Collection, Map or null). Symbol is also returned as a shallow copy.
    */
    private Object cloneObject(Object obj) {
        if (obj instanceof String || obj instanceof Symbol) {
            return obj;
        } else if (obj instanceof Collection) {
            return cloneCollection((Collection)obj);
        } else if (obj instanceof Map) {
            return cloneMap((Map)obj);
        } else if (obj == null) {
            // for some keys in the maps, values are null
            return null;
        } else {
            System.err.println(
                    "unhandled Object type, fix me in Annotation.java");
            return null;
        }
    }

    /**
    * returns the deep copy of the given collection 
    */
    @SuppressWarnings("unchecked")
    private LinkedList cloneCollection(Collection c) {
        Iterator iter = c.iterator();
        LinkedList list = new LinkedList();
        while (iter.hasNext()) {
            Object val = iter.next();
            list.add(cloneObject(val));
        }
        return list;
    }

    /**
    * Attaches a link from this annotation to the specified IR.
    * Declaration and Statement implement the Annotatable interface.
    * @param ir the associated Cetus IR.
    */
    public void attach(Annotatable ir) {
        this.ir = ir;
    }

    /**
    * Removes this annotation from its associated annotatable object.
    */
    public void detach() {
        ir.getAnnotations().remove(this);
        ir = null;
    }

    /**
    * Sets the skip_print field.
    */
    public void setSkipPrint(boolean skip_print) {
        this.skip_print = skip_print;
    }

    /**
    * Returns the annotatable object that contains this annotation.
    */
    public Annotatable getAnnotatable() {
        return ir;
    }

    /**
    * Disables printing of the specified annotation type attached at the given
    * annotatable object.
    * @param at the annotatable object of interest (statement or declaration).
    * @param type the annotation type to be hidden.
    */
    public static <T extends Annotation> void
            hideAnnotations(Annotatable at, Class<T> type) {
        for (T annotation : at.getAnnotations(type)) {
            annotation.setSkipPrint(true);
        }
    }

}

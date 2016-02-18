package cetus.hir;

/**
* Thrown when an action is performed on an IR object and
* that action requires the object to be the child of
* another object, but that is not the case.
*/
public class NotAChildException extends RuntimeException {

    private static final long serialVersionUID = 3479L;

    public NotAChildException() {
        super();
    }
    
    public NotAChildException(String message) {
        super(message);
    }

}

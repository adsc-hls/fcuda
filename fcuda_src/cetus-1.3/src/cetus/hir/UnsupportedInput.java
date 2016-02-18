package cetus.hir;


/**
 * Thrown when an action is performed on an IR object and
 * that action requires the object to be the child of
 * another object, but that is not the case.
 */
public class UnsupportedInput extends RuntimeException
{
  private static final long serialVersionUID = 1; /* avoids gcc 4.3 warning */
  public UnsupportedInput()
  {
    super();
  }

  public UnsupportedInput(String message)
  {
    super(message);
  }
}

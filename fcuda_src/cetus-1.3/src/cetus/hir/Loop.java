package cetus.hir;

/**
* Any class implementing this interface has the properties of a loop.
*/
public interface Loop extends Traversable {

    /**
    * Returns the statement that is the body of the loop.
    *
    * @return the body of the loop.
    */
    Statement getBody();

    /**
    * Returns the expression that determines the duration of the loop.
    *
    * @return the condition that is tested each iteration.
    */
    Expression getCondition();


  /** *AP*
   * Sets the loops condition to the argument expression
   */
  void setCondition(Expression cond);

}

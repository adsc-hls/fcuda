package cetus.hir;

/**
* Any class implementing this interface has the properties of what we 
* call an intrinsic function, it has a specific meaning and is 
* represented as an Expression in the IR.
*/
public interface Intrinsic {

    /**
    * Returns the expression.
    *
    * @return the expression that is the operand to which this intrinsic
    *       function is being applied as an operator.
    */
    public Expression getExpression();

}
/* Look at FunctionCall.java -> getArguments();
*/

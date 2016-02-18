package cetus.analysis;

import cetus.hir.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Class CallSite represents a container that mangages the source of a top-down
 * problem and the target of a bottom-up problem.
 */
public class CallSite {
    // Exceptions.
    private static final int FUNCTION_POINTER = 1;
    private static final int LIBRARY_CALL = 2;
    private static final int VARIABLE_ARG_LIST = 4;

    // The FunctionCall object matching with this call site.
    private FunctionCall fcall;

    // The calling node and the called node.
    private IPANode caller, callee;

    // List of actual arguments.
    private List<Expression> arguments;

    // List of normalized arugments.
    private List<Expression> norm_arguments;

    // Normalized arguments stored in a compound statement.
    private CompoundStatement temp_assignments;

    // In/Out data domain.
    private Domain in, out;

    // Unique ID.
    private int id;

    // Exception code.
    private int exception;

    /**
    * Constructs a call site with the given function call, caller node, and
    * callee node.
    */
    public CallSite(int id, FunctionCall fcall, IPANode caller,
                    IPANode callee) {
        this.id = id;
        this.fcall = fcall;
        this.caller = caller;
        this.callee = callee;
        exception = 0;
        clean();
        buildArgumentList();
        // normalizeArguments(); --> normalization is done by request.
    }

    /**
    * Returns the statement containing the function call.
    * @return the ancestor statement of the function call.
    */
    public Statement getStatement() {
        Statement ret = fcall.getStatement();
        // initial statement in a for statement should return the for statement.
        if (ret.getParent() instanceof ForLoop) {
            ret = (Statement)ret.getParent();
        }
        return ret;
    }

    /**
    * Returns the called node.
    * @return the callee node.
    */
    public IPANode getCallee() {
        return callee;
    }

    /**
    * Returns the calling node.
    * @return the caller node.
    */
    public IPANode getCaller() {
        return caller;
    }

    /**
    * Returns the function call associated with this call site.
    * @return the function call.
    */
    public FunctionCall getFunctionCall() {
        return fcall;
    }

    /**
    * Returns the ID of this call site.
    */
    public int getID() {
        return id;
    }

    /**
    * Returns the IN data.
    */
    @SuppressWarnings("unchecked")
        public <T extends Domain> T in() {
        return (T)in;
    }

    /**
    * Returns the OUT data.
    */
    @SuppressWarnings("unchecked")
        public <T extends Domain> T out() {
        return (T)out;
    }

    /**
    * Sets the IN data with the given Domain object.
    */
    public void in(Domain domain) {
        in = domain;
    }

    /**
    * Sets the OUT data with the given Domain object.
    */
    public void out(Domain domain) {
        out = domain;
    }

    /**
    * Removes the IN/OUT data.
    */
    public void clean() {
        in = NullDomain.getNull();
        out = NullDomain.getNull();
    }

    /**
    * Checks if the function takes variable argument list.
    */
    public boolean containsVarArg() {
        return ((exception & VARIABLE_ARG_LIST) != 0);
    }

    /**
    * Checks if there is a matching function body for the call site.
    */
    public boolean isLibraryCall() {
        return ((exception & LIBRARY_CALL) != 0);
    }

    /**
    * Checks if the function call is using a function pointer variable.
    */
    public boolean containsFunctionPointer() {
        return ((exception & FUNCTION_POINTER) != 0);
    }

    /**
    * Returns the string name of the function name.
    */
    public String getName() {
        return fcall.getName().toString();
    }

    // Builds parameter mapping.
    private void buildArgumentList() {
        arguments = new ArrayList<Expression>(4);
        for (int i = 0; i < fcall.getNumArguments(); i++) {
            arguments.add(fcall.getArgument(i));
        }
        // Catches exceptional cases.
        Symbol s = SymbolTools.getSymbolOf(fcall.getName());
        if (s instanceof NestedDeclarator &&
            ((NestedDeclarator)s).isProcedure()) {
            PrintTools.printlnStatus(0,
                    "[WARNING] function pointer in ", fcall);
            exception |= FUNCTION_POINTER;
        } else if (callee == null) {
            exception |= LIBRARY_CALL;
        } else if (callee.containsVarArg()) {
            exception |= VARIABLE_ARG_LIST;
            PrintTools.printlnStatus(0,
                    "[WARNING] variable argument list ", fcall);
        }
    }

    /**
    * Performs normalization of the non-identifier arguments. It creates a
    * temporary compound statement filled with the temporary assignments to set
    * of identifiers that takes the original arugments as RHS.
    */
    @SuppressWarnings("unchecked")
    protected void normalizeArguments() {
        if (exception != 0) {    // no normalization is possible.
            return;
        }
        temp_assignments = new CompoundStatement();
        norm_arguments = new ArrayList<Expression>(4);
        for (int i = 0; i < arguments.size(); i++) {
            Symbol param = getParameters().get(i);
            Expression arg = Symbolic.simplify(arguments.get(i));
            // Allows normalization of identifers which are global array names.
            if (arg instanceof Identifier) {
                Symbol base = ((Identifier)arg).getSymbol();
                if (SymbolTools.isGlobal(base) && SymbolTools.isArray(base)) {
                    arg = new UnaryExpression(
                            UnaryOperator.ADDRESS_OF,
                            new ArrayAccess(arg.clone(),new IntegerLiteral(0)));
                }
            }
            Expression new_arg = arg;
            if (!(arg instanceof Literal || arg instanceof Identifier)) {
                arg = normalize(arg, param);
                List type_spec = param.getTypeSpecifiers();
                List array_spec = param.getArraySpecifiers();
                // Assumes there is only one element in array_spec. 
                if (!array_spec.isEmpty()) {
                    type_spec.add(PointerSpecifier.UNQUALIFIED);
                }
                new_arg = SymbolTools.getTemp(temp_assignments,type_spec,"arg");
                Expression assign = new AssignmentExpression(
                        new_arg, AssignmentOperator.NORMAL, arg.clone());
                temp_assignments.addStatement(new ExpressionStatement(assign));
            }
            norm_arguments.add(new_arg);
        }
    }

    /** Normalization for points-to analysis. */
    private static Expression normalize(Expression arg, Symbol param) {
        Expression ret = arg.clone();   // default behavior.
        // Converts array accesses to address-of expressions.
        if (ret instanceof ArrayAccess) {
            ArrayAccess array = (ArrayAccess)ret;
            // Normalize the array name.
            Expression name = normalize(array.getArrayName(), null);
            Symbol base = SymbolTools.getSymbolOf(name);
            if (base != null && param != null &&
                SymbolTools.isPointerParameter(param)) {
                // case 1: the base symbol has an array specifier.
                // --> - keeps the array indices while adding trailing [0].
                //     - converts to address-of expression.
                //     - adds dereference operator if base is a formal parameter
                if (SymbolTools.isArray(base)) {
                    // Adds a trailing subscript "[0]" intentionally.
                    array.addIndex(new IntegerLiteral(0));
                    ret = new UnaryExpression(UnaryOperator.ADDRESS_OF, ret);
                    // Formal parameter is normalized: a[10] to (*a)[10].
                    // This conversion is used only internally (not legal in C).
                    if (SymbolTools.isPointerParameter(base)) {
                        array.getArrayName().swapWith(
                                new UnaryExpression(UnaryOperator.DEREFERENCE,
                                                    name.clone()));
                    }
                // case 2: the base symbol does not have an array specifier.
                // --> just take the base object while converting a subscript
                // to a dereference.
                } else {
                    ret = name;
                    for (int i = 0; i < array.getNumIndices(); i++) {
                        ret = new UnaryExpression(UnaryOperator.DEREFERENCE,
                                                  ret.clone());
                    }
                }
            } else {             // just normalizes the array name.
                array.getArrayName().swapWith(name);
            }
        // Removes pointer access and adds trailing dummy index for pointer
        // type.
        } else if (ret instanceof AccessExpression) {
            AccessExpression access = (AccessExpression) ret;
            // POINTER_ACCESS to MEMBER_ACCESS
            if (access.getOperator() == AccessOperator.POINTER_ACCESS) {
                // Normalize the LHS.
                Expression lhs = normalize(access.getLHS(), null);
                ret = new AccessExpression(new UnaryExpression(
                        UnaryOperator.DEREFERENCE, lhs.clone()),
                        AccessOperator.MEMBER_ACCESS,
                        access.getRHS().clone());
            }
            // Pointer type to address-of expression.
            if (param != null && SymbolTools.isPointerParameter(param)) {
                ret = new UnaryExpression(UnaryOperator.ADDRESS_OF,
                        new ArrayAccess(ret.clone(), new IntegerLiteral(0)));
            }
        // Just normalize the expression child.
        } else if (ret instanceof UnaryExpression) {
            UnaryExpression ue = (UnaryExpression)ret;
            ue.setExpression(normalize(ue.getExpression(), null));
        // Tries to convert simple pointer arithmetic to array access.
        } else if (ret instanceof BinaryExpression) {
            BinaryExpression be = (BinaryExpression)ret;
            Expression lhs = normalize(be.getLHS(), null);
            Expression rhs = normalize(be.getRHS(), null);
            if (param != null && SymbolTools.isPointerParameter(param) &&
                be.getOperator() == BinaryOperator.ADD) {
                if (isPointerArithmeticOperand(lhs, rhs)) {
                    ret = new UnaryExpression(UnaryOperator.ADDRESS_OF,
                            new ArrayAccess(rhs.clone(), lhs.clone()));
                } else if (isPointerArithmeticOperand(rhs, lhs)) {
                    ret = new UnaryExpression(UnaryOperator.ADDRESS_OF,
                            new ArrayAccess(lhs.clone(), rhs.clone()));
                }
            } else {
                ret = new BinaryExpression(lhs.clone(), be.getOperator(),
                                           rhs.clone());
            }
        // Type cast is discarded.
        } else if (ret instanceof Typecast) {
            ret = (Expression)ret.getChildren().get(0);
        }
        return ret;
    }

    /** Checks if the given expression is a possible pointer arithmetic. */
    private static boolean
            isPointerArithmeticOperand(Expression lhs, Expression rhs) {
        return ((lhs instanceof IntegerLiteral ||
                 lhs instanceof Identifier &&
                 SymbolTools.isInteger(SymbolTools.getSymbolOf(lhs))) &&
                (rhs instanceof Identifier &&
                 (SymbolTools.isArray(SymbolTools.getSymbolOf(rhs)) ||
                  SymbolTools.isPointer(SymbolTools.getSymbolOf(rhs)))));
    }

    /**
    * Returns a list of arguments with Identifier type.
    */
    public List<Expression> getIDArguments() {
        List<Expression> ret = new ArrayList<Expression>(4);
        for (int i = 0; i < arguments.size(); i++) {
            Expression arg = arguments.get(i);
            if (arg instanceof Identifier) {
                ret.add(arg);
            }
        }
        return ret;
    }

    /**
    * Returns the list of the call site's arguments.
     */
    public List<Expression> getArguments() {
        return arguments;
    }

    /**
    * Returns the list of the callee's parameters.
    */
    public List<Symbol> getParameters() {
        if (callee != null) {
            return callee.getParameters();
        } else {
            return null;
        }
    }

    /**
    * Converts the given argument to the corresponding parameter. Because of the
    * equals method of the expression, it returns the first parameter whose
    * argument matches the given argument.
    */
    public Symbol argumentToParameter(Expression arg) {
        int id = arguments.indexOf(arg);
        return (id < 0) ? null : getParameters().get(id);
    }

    /**
    * Converts the given parameter to the corresponding argument.
    */
    public Expression parameterToArgument(Symbol param) {
        int id = getParameters().indexOf(param);
        return (id < 0) ? null : arguments.get(id);
    }

    /**
    * Returns the list of temporary assignments stored in a compound statement.
    */
    public CompoundStatement getTempAssignments() {
        return temp_assignments;
    }

    /**
    * Returns the list of normalized identifier arguments.
    */
    public List<Expression> getNormArguments() {
        return norm_arguments;
    }

    /**
    * Returns the string dump of this call site.
    * @return string dump.
    */
    public String toString() {
        StringBuilder sb = new StringBuilder(80);
        sb.append(id).append(" ").append(caller.getName()).append("-->");
        sb.append(fcall.getName());
        if (PrintTools.getVerbosity() >= 3) {
            sb.append(" (").append(System.identityHashCode(fcall)).append(")");
        }
        if (callee == null) {
            sb.append(" <lib>");
        } else if (temp_assignments != null) {
            // normalized arguments
            String args = temp_assignments.toString().replaceAll("\n", " ");
            sb.append(" ").append(args);
        }
        if (PrintTools.getVerbosity() >= 3) {
            sb.append("\n        parent    = ").append(fcall.getParent());
            sb.append("\n        fcall     = ").append(fcall);
            sb.append("\n        args      = ").append(arguments);
            sb.append("\n        params    = ").append(getParameters());
            sb.append("\n        nargs     = ").append(norm_arguments);
            sb.append("\n        exception = ").append(exception);
        }
        return sb.toString();
    }
}

package cetus.transforms;

import cetus.analysis.IPPointsToAnalysis;
import cetus.hir.*;
import java.util.List;

/**
* This normalization pass is intended to simplify return statements for the
* Procedures in the program. Return statements with no return expressions are
* added at the end of the procedure if a return statement doesn't exist already.
* For all return statements in the procedure, the return expressions are
* assigned to a standard return variable and are replaced in the return
* statement with this standard return variable. Especially useful for statement
* based analyses and simplification of interprocedural analyses.
*/
public class NormalizeReturn extends ProcedureTransformPass {

    public NormalizeReturn(Program program) {
        super(program);
        disable_invalidation = true;
    }

    public String getPassName() {
        return "[NormalizeReturn]";
    }
  
    @SuppressWarnings("unchecked")
    public void transformProcedure(Procedure proc) {
        Identifier return_id = null;
        CompoundStatement cstmt = proc.getBody();
        // Get the last statement of the procedure
        List<Traversable> children = cstmt.getChildren();
        // Empty procedure body will have "null" last_statement.
        Statement last_statement = null;
        for (int i = children.size()-1; i >= 0; i--) {
            if (!(last_statement instanceof AnnotationStatement)) {
                last_statement = (Statement)children.get(i);
                break;
            }
        }
        // Irrespective of the return type, if the procedure 
        // doesn't already contain a return statement, insert one.
        // This does not change the semantics as the return value 
        // for a procedure with a valid return type but no return 
        // statement is undefined in the first place
        if (!(last_statement instanceof ReturnStatement)) {
            ReturnStatement return_statement = new ReturnStatement();
            if (children.isEmpty()) {
                cstmt.addStatement(return_statement);
            } else {
                cstmt.addStatementAfter(last_statement, return_statement);
            }
        }
        // If procedure has a return type other than void, then 
        // create a standard return variable of return type for the
        // procedure and insert a declaration for it in the procedure body
        // Now get all return statements for the procedure and simplify
        DFIterator<ReturnStatement> iter =
                new DFIterator<ReturnStatement>(cstmt, ReturnStatement.class);
        while (iter.hasNext()) {
            ReturnStatement ret_stmt = iter.next();
            Expression ret_expr = ret_stmt.getExpression();
            List return_type = proc.getReturnType();
            // static type is not necessary for the temporary variable.
            if (return_type.isEmpty()) {
                return_type.add(Specifier.INT); // implicit return type
            }
            while (return_type.remove(Specifier.STATIC));
            while (return_type.remove(Specifier.EXTERN));
            while (return_type.remove(Specifier.INLINE));
            boolean is_void = return_type.size() == 1 &&
                              return_type.get(0) == Specifier.VOID;
            if (ret_expr == null && !is_void) {
                if (return_id == null) {
                    return_id =
                            SymbolTools.getTemp(cstmt, return_type, "_ret_val");
                    IPPointsToAnalysis.return_vars.add(return_id.getSymbol());
                }
                ret_stmt.setExpression(return_id);
            } else if (ret_expr != null && !(ret_expr instanceof Identifier)) {
                // Create return variable if it hasn't already been created
                // The return expression will be non-null only if there 
                // exists a return type on the procedure
                if (return_id == null) {
                    return_id =
                            SymbolTools.getTemp(cstmt, return_type, "_ret_val");
                    IPPointsToAnalysis.return_vars.add(return_id.getSymbol());
                }
                // Use clone
                AssignmentExpression new_assign = new AssignmentExpression(
                        return_id.clone(),
                        AssignmentOperator.NORMAL,
                        ret_expr.clone());
                ExpressionStatement assign_stmt =
                        new ExpressionStatement(new_assign);
                // Insert the new assignment statement right before the return
                // statement
                CompoundStatement parent =
                        (CompoundStatement)ret_stmt.getParent();
                parent.addStatementBefore(ret_stmt, assign_stmt);
                // Replace the return expression in the return statement
                // with the new return var
                ret_expr.swapWith(return_id.clone());
            }
        }
    }
}

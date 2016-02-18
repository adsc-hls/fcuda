package cetus.transforms;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.LinkedList;
import java.util.ArrayList;
import cetus.analysis.RangeAnalysis;
import cetus.analysis.RangeDomain;
import cetus.exec.Driver;
import cetus.hir.*;

/**
* This transformation pass detects unreachable code due to branches that can be
* evaluated at compile time. This is a complement transformation to the
* dead-code elimination based on data dependences.
*/
public class BranchEliminator extends TransformPass {

    /** Name of the pass */
    private static final String pass_name = "[BranchEliminator]";

    /** Single empty range for cases without any available ranges */
    private static final RangeDomain empty_range = new RangeDomain();

    /** Option for branch elimination */
    private int option;

    /** Result of transformation  */
    private int[] result = new int[] {0, 0, 0, 0, 0, 0};
    private static final int IF     = 0;
    private static final int FOR    = 1;
    private static final int WHILE  = 2;
    private static final int DO     = 3;
    private static final int SWITCH = 4;
    private static final int CASE   = 5;

    /** Constructs a new branch eliminator */
    public BranchEliminator(Program program) {
        super(program);
        try {
            option =
                Integer.parseInt(Driver.getOptionValue("teliminate-branch"));
        } catch (NumberFormatException ex) {
            option = 0;
        }
    }

    /** Returns the pass name */
    public String getPassName() {
        return pass_name;
    }

    /**
    * Performs branch elimination until there is no more opportunities
    * remaining.
    */
    public void start() {
        DFIterator<Procedure> iter =
                new DFIterator<Procedure>(program, Procedure.class);
        iter.pruneOn(Procedure.class);
        while (iter.hasNext()) {
            Procedure proc = iter.next();
            processCompound(proc.getBody());
            normalizeCompound(proc);
        }
        PrintTools.printlnStatus(1, pass_name, "Removed Branches:");
        PrintTools.printlnStatus(1, pass_name, "IF    :", result[IF]);
        PrintTools.printlnStatus(1, pass_name, "FOR   :", result[FOR]);
        PrintTools.printlnStatus(1, pass_name, "WHILE :", result[WHILE]);
        PrintTools.printlnStatus(1, pass_name, "DO    :", result[DO]);
        PrintTools.printlnStatus(1, pass_name, "SWITCH:", result[SWITCH]);
        PrintTools.printlnStatus(1, pass_name, "CASE  :", result[CASE]);
    }

    /**
    * Performs branch elimination within the specified compound statement.
    * Actual transformation occurs within a compound statement while the child
    * candidate statements are analyzed for elimination.
    * @param cstmt the compound statement to be transformed.
    * @return the number of eliminations within {@code cstmt}.
    */
    private void processCompound(Statement cstmt) {
        if (!(cstmt instanceof CompoundStatement)) {
            return;
        }
        List<Traversable> children = cstmt.getChildren();
        List<Statement> swap_pairs = new LinkedList<Statement>();
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Statement new_stmt = null;
            Traversable child = children.get(i);
            if (child instanceof IfStatement) {
                IfStatement if_stmt = (IfStatement)child;
                new_stmt = processIf(if_stmt);
                if (new_stmt == null) {
                    Statement then_stmt = if_stmt.getThenStatement();
                    Statement else_stmt = if_stmt.getElseStatement();
                    processCompound(then_stmt);
                    if (else_stmt != null) {
                        processCompound(else_stmt);
                    }
                }
            } else if (child instanceof Loop) {
                Loop loop = (Loop)child;
                new_stmt = processLoop(loop);
                if (new_stmt == null) {
                    processCompound(loop.getBody());
                }
            } else if (child instanceof SwitchStatement) {
                SwitchStatement switch_stmt = (SwitchStatement)child;
                new_stmt = processSwitch(switch_stmt);
            } else if (child instanceof CompoundStatement) {
                processCompound((Statement)child);
            }
            if (new_stmt != null) {
                swap_pairs.add((Statement)child);
                swap_pairs.add(new_stmt);
            }
        }
        while (!swap_pairs.isEmpty()) {
            Statement old_stmt = swap_pairs.remove(0);
            Statement new_stmt = swap_pairs.remove(0);
            if (option > 1) {
                // Annotates the new statement with the old statement.
                String comment = "Eliminated Branch:" + PrintTools.line_sep;
                if (option == 2) {
                    comment += old_stmt.getClass().getName();
                } else {
                    comment += old_stmt.toString();
                }
                new_stmt.annotateBefore(new CommentAnnotation(comment));
            }
            old_stmt.swapWith(new_stmt);
        }
    }

    /**
    * Analyzes {@link IfStatement} type for branch elimination. The code section
    * that follows the "false" condition is unreachable if it does not contain
    * any branch target such as a label.
    * @param if_stmt the if statement to be analyzed.
    * @param num_elims the number of eliminations within {@code if_stmt}.
    * @return the statement that can replace {@code if_stmt}.
    */
    private Statement processIf(IfStatement if_stmt) {
        Statement ret = null;
        Expression control = if_stmt.getControlExpression();
        if (DataFlowTools.getDefList(control).size() > 0 ||
            IRTools.containsFunctionCall(control)) {
            return ret;
        }
        Statement then_stmt = if_stmt.getThenStatement();
        Statement else_stmt = if_stmt.getElseStatement();
        RangeDomain range = RangeAnalysis.query(if_stmt);
        if (range == null) {
            range = empty_range;
        }
        int value = range.evaluateLogic(control);
        if (value == 0) {
            if (!IRTools.containsClass(then_stmt, Label.class)) {
                if (else_stmt == null) {
                    ret = new NullStatement();
                } else {
                    // Process the remaining parts before cloning.
                    processCompound(else_stmt);
                    else_stmt.setParent(null);
                    ret = else_stmt;
                }
                result[IF]++;
            }
        } else if (value == 1) {
            if (else_stmt == null ||
                !IRTools.containsClass(else_stmt, Label.class)) {
                // Process the remaining parts before cloning.
                processCompound(then_stmt);
                then_stmt.setParent(null);
                ret = then_stmt;
                result[IF]++;
            }
        }
        return ret;
    }

    /**
    * Analyzes {@link Loop} type for branch elimination.
    * @param loop the loop statement to be analyzed.
    * @return the statement that can replace {@code loop}.
    */
    private Statement processLoop(Loop loop) {
        Statement ret = null;
        Expression condition = loop.getCondition();
        if (condition == null || IRTools.containsClass(loop, Label.class)) {
            return ret;
        }
        RangeDomain range = new RangeDomain();
        if (loop instanceof DoLoop) {
            // For a do while loop, use the range domain associated with the
            // last executable statement within the loop body.
            List<Traversable> children = loop.getBody().getChildren();
            Statement last_stmt = null;
            for (int i = children.size()-1; i >= 0; i++) {
                Traversable child = children.get(i);
                if (!(child instanceof AnnotationStatement) &&
                    !(child instanceof DeclarationStatement)) {
                    last_stmt = (Statement)child;
                    break;
                }
            }
            if (last_stmt != null) {
                range.intersectRanges(RangeAnalysis.query(last_stmt));
            }
        } else {
            range.intersectRanges(RangeAnalysis.query((Statement)loop));
        }
        if (loop instanceof ForLoop) {
            Statement init_stmt = ((ForLoop)loop).getInitialStatement();
            if (init_stmt instanceof ExpressionStatement) {
                range.removeRangeWith(DataFlowTools.getDefSymbol(init_stmt));
                range.intersectRanges(RangeAnalysis.getRangeDomain(init_stmt));
            }
        }
        int value = range.evaluateLogic(condition);
        if (value == 0) {
            if (loop instanceof DoLoop) { // needs to return the body
                Statement body = loop.getBody();
                // Process the body before cloning.
                processCompound(body);
                body.setParent(null);
                ret = body;
                result[DO]++;
            } else if (loop instanceof ForLoop) {
                Statement init_stmt = ((ForLoop)loop).getInitialStatement();
                if (init_stmt != null) {
                    init_stmt.setParent(null);
                    ret = init_stmt;
                } else {
                    ret = new NullStatement();
                }
                result[FOR]++;
            } else {
                ret = new NullStatement();
                result[WHILE]++;
            }
        }
        return ret;
    }

    /**
    * Analyzes {@link SwitchStatement} type for branch elimination.
    */
    private Statement processSwitch(SwitchStatement switch_stmt) {
        Statement ret = null;
        Expression switch_expr = switch_stmt.getExpression();
        Statement body = switch_stmt.getBody();
        // Process the body first before analyzing the swtich branches.
        processCompound(body);
        if (IRTools.containsClass(body, Label.class)) {
            return ret;
        }
        // Eligibility test: all "case", "break", and "default" statements
        // should be direct children of the body.
        DFIterator<Statement> iter =
                new DFIterator<Statement>(body, Statement.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            if (stmt instanceof BreakStatement ||
                stmt instanceof Case || stmt instanceof Default) {
                SwitchStatement parent_switch = IRTools.getAncestorOfType(
                        stmt, SwitchStatement.class);
                if (parent_switch == switch_stmt &&
                    !body.getChildren().contains(stmt)) {
                    return ret;
                }
            }
        }
        // At this point, the switch statement has a simple structure where
        // all case/default/break statements are direct children of the body.
        List<Statement> true_stmts = new LinkedList<Statement>();
        List<Statement> false_stmts = new LinkedList<Statement>();
        RangeDomain range = RangeAnalysis.query(switch_stmt);
        if (range == null) {
            range = empty_range;
        }
        // Evaluates each jump targets while collecting statements to be
        // removed or to remain.
        List<Traversable> children = body.getChildren();
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = children.get(i);
            if (child instanceof Case) {
                Case case_stmt = (Case)child;
                int value = range.evaluateLogic(
                        Symbolic.eq(switch_expr, case_stmt.getExpression()));
                if (value == 0) { // Unreachable target
                    false_stmts.add(case_stmt);
                    for (int j = i+1; j < children_size; j++) {
                        Statement stmt = (Statement)children.get(j);
                        if (stmt instanceof Default || stmt instanceof Case) {
                            break;
                        } else if (!false_stmts.contains(stmt)) {
                            false_stmts.add(stmt);
                        }
                    }
                } else if (value == 1) { // The only reachable target
                    for (int j = i+1; j < children_size; j++) {
                        Statement stmt = (Statement)children.get(j);
                        if (stmt instanceof BreakStatement) {
                            break;
                        } else if (!(stmt instanceof Case) &&
                                    !(stmt instanceof Default)) {
                            true_stmts.add(stmt);
                        }
                    }
                }
            }
        }
        if (!true_stmts.isEmpty()) {
            List<Traversable> children_copy =
                    new ArrayList<Traversable>(children);
            for (int i = 0; i < children_size; i++) {
                Statement stmt = (Statement)children_copy.get(i);
                if (!(stmt instanceof DeclarationStatement) &&
                    !true_stmts.contains(stmt)) {
                    stmt.detach();
                }
            }
            ret = body;
            result[SWITCH]++;
        } else if (!false_stmts.isEmpty()) {
            while (!false_stmts.isEmpty()) {
                Statement stmt = false_stmts.remove(0);
                stmt.detach();
                if (stmt instanceof Case) {
                    result[CASE]++;
                }
            }
        }
        return ret;
    }

    /**
    * Removes unnecessary compound statements that do not have any variable
    * declaration. Note that this operation may remove some compound statements
    * that existed in the original source code.
    * @param t the traversable object to be transformed.
    */ 
    private void normalizeCompound(Traversable t) {
        List<CompoundStatement> cstmts = (new DFIterator<CompoundStatement>(
                t, CompoundStatement.class)).getList();
        if (t instanceof CompoundStatement) {
            cstmts.remove(0);
        }
        for (int i = cstmts.size()-1; i >= 0; i--) {
            CompoundStatement cstmt = cstmts.get(i);
            if (cstmt.getParent() instanceof CompoundStatement &&
                !IRTools.containsClass(cstmt, Declaration.class)) {
                CompoundStatement parent = (CompoundStatement)cstmt.getParent();
                List<Traversable> children = cstmt.getChildren();
                while (!children.isEmpty()) {
                    Statement child = (Statement)children.get(0);
                    child.detach();
                    parent.addStatementBefore(cstmt, child);
                }
                cstmt.detach();
            }
        }
    }
}

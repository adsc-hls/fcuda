package cetus.transforms;

import cetus.analysis.LoopTools;
import cetus.hir.*;

import java.util.LinkedList;
import java.util.List;

/**
 * LoopProfile inserts timers around loops following the selection strategy
 * specified by "select" field.
 */
public class LoopProfiler extends TransformPass {

    /** Profiling option */
    private int strategy;
    /**
     * Profiling strategies:
     * EVERY         every for loop
     * OUTER         outermost for loop
     * EVERY_OMP_PAR every omp parallel region
     * OUTER_OMP_PAR outermost omp parallel region
     * EVERY_OMP_FOR every omp for loop
     * OUTER_OMP_FOR outermost omp for loop
     */
    private static final int EVERY = 1;
    private static final int OUTER = 2;
    private static final int EVERY_OMP_PAR = 3;
    private static final int OUTER_OMP_PAR = 4;
    private static final int EVERY_OMP_FOR = 5;
    private static final int OUTER_OMP_FOR = 6;

    /**
     * Constructs a new LoopProfile object from the specified program and
     * performs profiling. It collects information such as total number of
     * procedures, maximum number of loops per procedure, main procedure, and
     * main translation unit for code generation.
     */
    public LoopProfiler(Program prog) {
        super(prog);
        strategy = Integer.valueOf(
                cetus.exec.Driver.getOptionValue("profile-loops")).intValue();
        //disable_protection = true;
    }

    public String getPassName() {
        return "[LoopProfiler]";
    }

    public void start() {
        if (strategy > 0) {
            int event_num = 0;
            LoopTools.addLoopName(program);
            List<Statement> stmts = new LinkedList<Statement>();
            collectCandidates(program, stmts);
            for (Statement stmt : stmts) {
                String loop_name = LoopTools.getLoopName(stmt);
                if (loop_name == null) { // stmt is not a loop.
                    loop_name = "event#" + (event_num++);
                }
                CompoundStatement parent =
                        IRTools.getAncestorOfType(stmt,CompoundStatement.class);
                Statement start = new AnnotationStatement();
                start.annotate(new PragmaAnnotation.Event(loop_name, "start"));
                Statement stop = new AnnotationStatement();
                stop.annotate(new PragmaAnnotation.Event(loop_name, "stop"));
                parent.addStatementBefore(stmt, start);
                parent.addStatementAfter(stmt, stop);
            }
        }
        TransformPass.run(new EventTimer(program));
    }

    /** Collect candidate statements following the specified option. */
    private void collectCandidates(Traversable t, List<Statement> stmts) {
        List<Traversable> children = t.getChildren();
        if (children == null || children.isEmpty()) {
            return;
        }
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = children.get(i);
            if (child instanceof Statement) {
                Statement stmt = (Statement)child;
                boolean contains_omp_par =
                        stmt.containsAnnotation(OmpAnnotation.class,"parallel");
                boolean contains_omp_for =
                        stmt.containsAnnotation(OmpAnnotation.class, "for");
                boolean contains_jump =
                        IRTools.containsClass(stmt, ReturnStatement.class)
                        || IRTools.containsClass(stmt, GotoStatement.class);
                boolean was_profiled = false;
                if (contains_jump) {
                    if (t instanceof Loop) {
                        PrintTools.printlnStatus(0,
                            "[WARNING] Skipping profiling",
                            "of the loop with \"return or goto\"");
                    }
                } else {
                    switch (strategy) {
                    case EVERY:
                    case OUTER:
                        if (stmt instanceof ForLoop) {
                            stmts.add(stmt);
                            was_profiled = true;
                        }
                        break;
                    case EVERY_OMP_PAR:
                    case OUTER_OMP_PAR:
                        if (contains_omp_par) {
                            stmts.add(stmt);
                            was_profiled = true;
                        }
                        break;
                    case EVERY_OMP_FOR:
                    case OUTER_OMP_FOR:
                        if (contains_omp_for) {
                            stmts.add(stmt);
                            was_profiled = true;
                        }
                        break;
                    default:
                    }
                }
                if (!was_profiled ||
                        (strategy != OUTER && strategy != OUTER_OMP_PAR
                        && strategy != OUTER_OMP_FOR)) {
                    collectCandidates(child, stmts);
                }
            } else if (!(child instanceof VariableDeclaration
                    || child instanceof ExpressionStatement)) {
                collectCandidates(child, stmts);
            }
        }
    }
}

package cetus.analysis;

import cetus.exec.Driver;
import cetus.hir.*;
import cetus.transforms.TransformPass;
import cetus.transforms.ReductionTransform;

import java.util.*;

/**
* Whole program analysis that uses data-dependence information to 
* internally annotate loops that are parallel
*/
public class LoopParallelizationPass extends AnalysisPass {

    // The level of parallelization requested from this pass
    private static final int PARALLELIZE_DISABLE_NESTED = 1;
    private static final int PARALLELIZE_LOOP_NEST = 2;

    // Store the level of parallelization required
    private int parallelization_level;

    // Enable/disable nested loop parallelism
    private boolean nested_parallelism;

    // Data structure for parallelization report.
    private Map<Loop, List<String>> report;

    // Flag for report generation.
    private boolean needs_report;

    // Pass name.
    private static final String pass_name = "[LoopParallelization]";

    /**
    * Constructs a new parallelization pass with the specified program.
    * @param program the program to be parallelized.
    */
    public LoopParallelizationPass(Program program) {
        super(program);
        parallelization_level = Integer.valueOf
            (Driver.getOptionValue("parallelize-loops")).intValue();
        // adjust the number if "report" is requested.
        if (parallelization_level > 2) {
            parallelization_level -= 2;
            needs_report = true;
        } else {
            needs_report = false;
        }
        report = new HashMap<Loop, List<String>>();
    }

    /**
    * Get Pass name
    */
    public String getPassName() {
        return pass_name;
    }

    /**
    * Start whole program loop parallelization analysis.
    */
    public void start() {
        // Implemented nested or non-nested parallelism as per user request
        switch (parallelization_level) {
        case PARALLELIZE_LOOP_NEST:
            nested_parallelism = true;
            parallelizeAllNests();
            break;
        case PARALLELIZE_DISABLE_NESTED:
            nested_parallelism = false;
            parallelizeAllNests();
            break;
        }
        reportParallelization();
        // Invokes reduction transformation.
        if (!nested_parallelism) {
            TransformPass.run(new ReductionTransform(program));
        }
    }

    /**
    * Performs parallelization of the loop nests in the program.
    */
    public void parallelizeAllNests() {    
        DFIterator<Loop> iter = new DFIterator<Loop>(program, Loop.class);
        iter.pruneOn(Loop.class);
        while (iter.hasNext()) {
            parallelizeLoopNest(iter.next());
        }
    }

    /**
    * Inserts cetus annotation if the given loop is proven to be parallel.
    */
    private void addCetusAnnotation(Loop loop, boolean parallel) {
        if (parallel &&
            Driver.isIncluded("parallelize-loops",
                    "Loop", LoopTools.getLoopName((Statement)loop))) {
            CetusAnnotation note = new CetusAnnotation();
            note.put("parallel", "true");
            ((Annotatable)loop).annotate(note);
        }
    }

    /**
    * Check if a specific loop in the program is parallel, irrespective of the
    * effects of parallelizing or serializing enclosing loops.
    * @param loop the for loop to check parallelism for based only on
    * dependence analysis
    * @return true if it is parallel.
    */
    @Deprecated
    private boolean checkParallel(Loop loop) {
        boolean is_parallel = false;
        boolean nest_eligible = false;
        DDGraph loop_graph = null;
        DDGraph pdg = program.getDDGraph();
        // Check eligibility of enclosed nest for dependence testing
        List<Loop> entire_nest = LoopTools.calculateInnerLoopNest(loop);
        for (Loop l : entire_nest) {
            nest_eligible = LoopTools.checkDataDependenceEligibility(l);
            if (nest_eligible == false)
                break;
        }
        if (nest_eligible==true) {
            // Check if scalar dependences might exist
            if(LoopTools.scalarDependencePossible(loop)==true)
                is_parallel=false;
            // Check if early exit break statement might exist
            else if (LoopTools.containsBreakStatement(loop)==true)
                is_parallel=false;
            // check if array loop carried dependences exist
            else if (pdg.checkLoopCarriedDependence(loop)==true) {
                // Also check if loop carried dependences might be
                // because of private or reduction or induction variables
                loop_graph = pdg.getSubGraph(loop);
                ArrayList<DDGraph.Arc> loop_carried_deps =
                    loop_graph.getLoopCarriedDependencesForGraph();
                for (DDGraph.Arc dep : loop_carried_deps) {
                    if (dep.isCarried(loop)) {
                        ArrayAccess dep_access =
                            dep.getSource().getArrayAccess();
                        Symbol dep_symbol =
                            SymbolTools.getSymbolOf((Expression)dep_access);
                        // Check if loop carried dependence is for private
                        // variable
                        if (LoopTools.isPrivate(dep_symbol, loop))
                            is_parallel = true;
                        // Check if loop carried dependence is for reduction
                        // variable
                        else if (LoopTools.isReduction(dep_symbol, loop))
                            is_parallel = true;
                        //else if (LoopTools.isInductionVariable(dep_symbol, l))
                        //    is_parallel = true;
                        else {
                            is_parallel = false;
                            break;
                        }
                    }
                    else
                        is_parallel = true;
                }
            }
            // No scalar or array dependences
            else
                is_parallel=true;
        }
        else
            is_parallel=false;

        return is_parallel;
    }

    /**
    * Using dependence information, parallelize the entire loop nest covered by
    * the enclosing loop. If an outer loop is found to be serial, serialize it
    * and eliminate all loop carried dependences originating from it, this will
    * in turn expose inner parallelism.
    * @param enclosing_loop the loop which encloses the nest to be parallelized.
    */
    private void parallelizeLoopNest(Loop enclosing_loop) {
        boolean is_parallel;
        DDGraph dependence_graph = program.getDDGraph();
        List<Loop> eligible_loops = LoopTools.
                extractOutermostDependenceTestEligibleLoops(enclosing_loop);
        for (int i = 0; i < eligible_loops.size(); i++) {
            Loop outer_loop = eligible_loops.get(i);
            DDGraph nest_ddgraph = dependence_graph.getSubGraph(outer_loop);
            List<Loop> contained_nest = 
                    LoopTools.calculateInnerLoopNest(outer_loop);
            // Records loops that are already scheduled for parallelization.
            List<Loop> scheduled = new ArrayList<Loop>(contained_nest.size());
            for (int j = 0; j < contained_nest.size(); j++) {
                boolean has_scheduled_outer_loop = false;
                Loop l = contained_nest.get(j);
                for (int k = 0; k < scheduled.size(); k++) {
                    if (IRTools.isAncestorOf(scheduled.get(k), l)) {
                        has_scheduled_outer_loop = true;
                        break;
                    }
                }
                // Does not analyze for parallelization if not necessary
                if (has_scheduled_outer_loop && !needs_report &&
                    !nested_parallelism) {
                    continue;
                }
                is_parallel = true;
                if (LoopTools.containsBreakStatement(l)) {
                    is_parallel = false;
                    addReport(l, "contains a break statement");
                    if (!needs_report) {
                        continue;
                    }
                }
                Set<Expression> scalar_deps =
                        LoopTools.collectScalarDependences(l);
                if (!scalar_deps.isEmpty()) {
                    is_parallel = false;
                    addReport(l, "contains scalar dependences on {" +
                            PrintTools.collectionToString(scalar_deps, ", ") +
                            "}");
                    if (!needs_report) {
                        continue;
                    }
                }
                Set<Symbol> array_deps = new HashSet<Symbol>();
                ArrayList<DDGraph.Arc> all_arcs = nest_ddgraph.getAllArcs();
                for (int k = 0; k < all_arcs.size(); k++) {
                    DDGraph.Arc row = all_arcs.get(k);
                    DependenceVector dv = row.getDependenceVector();
                    // If direction is loop carried
                    if (!dv.getDirectionVector().containsKey(l) ||
                        dv.getDirection(l) == DependenceVector.equal ||
                        dv.getDirection(l) == DependenceVector.nil) {
                        continue;
                    }
                    ArrayAccess src_access = row.getSource().getArrayAccess();
                    Symbol src_symbol =
                            SymbolTools.getSymbolOf((Expression)src_access);
                    ArrayAccess sink_access = row.getSink().getArrayAccess();
                    Symbol sink_symbol =
                            SymbolTools.getSymbolOf((Expression)sink_access);
                    // Check if loop carried dependence is for
                    // private variable or reduction variable.
                    // If not, must serialize this loop
                    boolean serialize;
                    // Alias relationship may incur different src and sink,
                    // which is not handled in both array privatization and
                    // reduction.
                    if (src_symbol == sink_symbol &&
                        (LoopTools.isPrivate(src_symbol, l) ||
                        LoopTools.isReduction(src_symbol, l))) {
                        serialize = false;
                    } else {
                        serialize = true;
                        array_deps.add(src_symbol);
                    }
                    if (serialize) {
                        is_parallel = false;
                    // Remove this dependence vector as serializing this loop
                    // will remove covered dependences (this direction will be
                    // < if the enclosing_loop passed into this loop is at the
                    // outermost level in the program. If not, it might be a >
                    // direction, but is assumed to be covered by an outer <
                    // direction and hence, the dependence vector can be deleted
                    // If the direction is any, there can be an equal direction
                    // as well and hence the row should not be deleted
                        if (dv.getDirection(l) != DependenceVector.any) {
                            all_arcs.remove(k--);
                        }
                    }
                }
                if (!array_deps.isEmpty()) {
                    addReport(l, "contains array dependences on {" +
                            PrintTools.collectionToString(array_deps, ", ") +
                            "}");
                }
                if (is_parallel) {
                    addReport(l, "is parallel");
                    if (nested_parallelism || !has_scheduled_outer_loop) {
                        addCetusAnnotation(l, true);
                        addReport(l, "is scheduled for parallelization");
                        scheduled.add(l);
                    }
                } else {
                    addReport(l, "is serial");
                }
            }
        }
    }

    /**
    * Adds a string entry in the report data structure.
    */
    protected void addReport(Loop loop, String text) {
        List<String> loop_report = report.get(loop);
        if (loop_report == null) {
            loop_report = new LinkedList<String>();
            report.put(loop, loop_report);
        }
        loop_report.add(text);
    }

    private void reportParallelization() {
        if (!needs_report) {
            return;
        }
        StringBuilder sb = new StringBuilder(400);
        String tag = "[AUTOPAR] ";
        DFIterator<ForLoop> iter =
                new DFIterator<ForLoop>(program, ForLoop.class);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(ExpressionStatement.class);
        while (iter.hasNext()) {
            ForLoop floop = iter.next();
            // Loop name
            sb.append(tag);
            sb.append("Loop is named ");
            sb.append(LoopTools.getLoopName(floop));
            sb.append(PrintTools.line_sep);
            // Any manual parallelization
            if (floop.containsAnnotation(OmpAnnotation.class, "for")) {
                sb.append(tag);
                sb.append("     was manually parallelized");
                sb.append(PrintTools.line_sep);
            }
            // Parallelization result
            List<String> floop_report = report.get(floop);
            if (floop_report != null) {
                for (String content : floop_report) {
                    sb.append(tag);
                    sb.append("     ");
                    sb.append(content);
                    sb.append(PrintTools.line_sep);
                }
            } else {
                sb.append(tag);
                sb.append("     is not eligible for parallelization");
                sb.append(PrintTools.line_sep);
            }
        }
        System.out.println(sb + "");
    }

    /**
    * Prints summary of loop parallelization pass.
    */
    /*
    private void reportParallelizationOld() {
        if ( PrintTools.getVerbosity() < 1 ) 
            return;
        String tag = "[PARALLEL REPORT] ";
        String separator = "";
        for ( int i=0; i<80-tag.length(); i++ ) separator += ":";
        StringBuilder legend = new StringBuilder(300);
        legend.append(tag+separator+"\n");
        legend.append(tag+"InputParallel: loop is parallel in the input program\n");
        legend.append(tag+"CetusParallel: loop is auto-parallelized\n");
        legend.append(tag+"NonCanonical : loop is not canonical\n");
        legend.append(tag+"NonPerfect   : loop is not perfect nest\n");
        legend.append(tag+"ControlFlow  : loop may exit prematually\n");
        legend.append(tag+"SymbolicStep : loop step is symbolic\n");
        legend.append(tag+"FunctionCall : loop contains function calls\n");
        legend.append(tag+"I/O          : loop contains I/O calls\n");
        legend.append(tag+separator+"\n");
        System.out.print(legend);

        String loop_name = "-";
        boolean omp_found = false, cetus_parallel_found = false;

        DepthFirstIterator iter = new DepthFirstIterator(program);
        while ( iter.hasNext() )
        {
            Object o = iter.next();

            if ( o instanceof ForLoop )
            {
                ForLoop for_loop = (ForLoop)o;
                StringBuilder out = new StringBuilder(80);
                out.append(LoopTools.getLoopName(for_loop));
                if ( for_loop.containsAnnotation(OmpAnnotation.class, "for") )
                    out.append(", InputParallel");
                if ( for_loop.containsAnnotation(CetusAnnotation.class, "parallel") )
                    out.append(", CetusParallel");
                if ( !LoopTools.isCanonical(for_loop) )
                    out.append(", NonCanonical");
                if ( !LoopTools.isPerfectNest(for_loop) )
                    out.append(", NonPerfect");
                if ( LoopTools.containsControlFlowModifier(for_loop) )
                    out.append(", ControlFlow");
                if ( !LoopTools.isIncrementEligible(for_loop) )
                    out.append(", SymbolicStep");
                if ( IRTools.containsClass(for_loop, FunctionCall.class) )
                    out.append(", FunctionCall");
                System.out.println(tag+out);
            }
        }
    }
    */
}

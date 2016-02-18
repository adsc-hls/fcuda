package cetus.codegen;

import cetus.exec.Driver;
import cetus.hir.*;
import cetus.analysis.LoopTools;
import java.util.List;
import java.util.Set;

/**
* This pass performs postprocessing of programs parallelized with OpenMP.
* Two methods, one with simple performance model and the other with guiding
* profiler, are provided. Both methods make use of "if" clause provided by
* the OpenMP specification.
* <p>
* <b>Model-based approach (starts with -profitable-omp=1)</b><br>
* Symbolically computes the workload of each loop by counting loop iteration
* and the child statements. If the workload is smaller than the given
* threshold, it is serialized either at compile-time (if the workload is
* integer literal) or at run-time (through OpenMP if clause).
* Following sub options are allowed through command line:
* <pre>
*     threshold=N  the threshold for serialization
*     costwhile=N  the iteration count for while-like loops
*     costfcall=N  the workload of a function call
*     runtime=0|1  when this value is zero OpenMP if clause is avoided
*   e.g., -profitable-omp=1:threshold=10000,costwhile=100,costfcall=100
* </pre>
* </p>
* <p>
* <b>Profile-based approach (starts with -profitable-omp=2)</b><br>
* This approach assumes the workload of a parallel loop does not change over
* multiple invocations of the loop. The basic idea is to use profiled
* runtime of a sequential execution and a parallel execution of a parallel
* loop and to decide whether the loop should be executed in parallel after
* the profiling. A parallel loop is executed in three different phases:
* <pre>
*     grace phase  : the loop is executed in parallel
*     profile phase: the loop is profiled several times both in serial and
*                    in parallel
*     stable phase : the loop is executed either in serial or in parallel
*                    based on the profiled information
* </pre>
*     Users can vary the behavior of each phase by specifying the following sub
*     options:
* <pre>
*     grace=N   : the length of grace phase (# invocations)
*     repeat=N  : the number of profiled invocations in serial/parallel
*     interval=N: the interval between successive profiling within the
*                 profile phase
*     usmin=N   : threshold for eager serialization without profiling in
*                 micro seconds 
*   e.g., -profitable-omp=2:grace=50,repeat=5,interval=10,usmin=10
* </pre>
* </p>
*/
public class ProfitableOMP extends CodeGenPass {

    private static final String pass_name = "[ProfitableOMP]";
    private static final int MODEL_BASED = 1;
    private static final int PROFILE_BASED = 2;
    private static final int HYBRID = 3;

    // option between model-based and profile-based approaches.
    private int option;

    // suboptions for model-based approach
    private int model_threshold = 10000; // workload threshold for serialization
    private int model_costwhile = 100;   // trip count for a while-like loop
    private int model_costfcall = 100;   // cost of a function call
    private int model_runtime   = 1;     // non-zero enables runtime generation
    // symbolic values for the parameters
    private Expression model_scostwhile;
    private Expression model_scostfcall;
    private static final Expression model_sone = new IntegerLiteral(1);
    private static final Expression model_szro = new IntegerLiteral(0);

    // suboptions for profile-based approach
    private int profile_grace    = 50; // grace invocations before profiling
    private int profile_interval = 10; // interval between repeated profiling
    private int profile_repeat   = 5;  // number of total profiling
    private int profile_usmin    = 10; // minimum workload in micro seconds
    private int profile_numevents;     // total number of profiled events
                                       //     (collected during this pass)
    private static final String profile_name = "cetusevents";

    /**
    * Constructs a new profitable code generator for OpenMP programs.
    * @param program the program to be processed.
    */
    public ProfitableOMP(Program program) {
        super(program);
        if (Driver.getOptionValue("profitable-omp") == null) {
            Driver.setOptionValue("profitable-omp", "1");
        }
        if (!parseModelBased() && !parseProfileBased() && !parseHybrid()) {
            PrintTools.printlnStatus(0, pass_name,
                    "Suboption is not recognized -- forcing default settings");
            option = 1;
        }
    }

    // parses the suboptions for model-based approach
    private boolean parseModelBased() {
        String[] values = Driver.getOptionValue("profitable-omp").split(":");
        if (!values[0].equals("1")) {
            return false;
        }
        option = MODEL_BASED;
        if (values.length == 2) {
            for (String suboption : values[1].split(",")) {
                String msg = "Ignoring unknown suboption \'" + suboption + "\'";
                int subvalue;
                String[] subvalues = suboption.split("=");
                if (subvalues.length != 2) {
                    PrintTools.printlnStatus(0, pass_name, msg);
                    continue;
                }
                try {
                    subvalue = Integer.parseInt(subvalues[1]);
                } catch (NumberFormatException ex) {
                    PrintTools.printlnStatus(0, pass_name, msg);
                    continue;
                }
                if (subvalues[0].equals("threshold")) {
                    model_threshold = subvalue;
                } else if (subvalues[0].equals("costwhile")) {
                    model_costwhile = subvalue;
                } else if (subvalues[0].equals("costfcall")) {
                    model_costfcall = subvalue;
                } else if (subvalues[0].equals("runtime")) {
                    model_runtime = subvalue;
                } else {
                    PrintTools.printlnStatus(0, pass_name, msg);
                }
            }
        }
        model_scostwhile = new IntegerLiteral(model_costwhile);
        model_scostfcall = new IntegerLiteral(model_costfcall);
        return true;
    }

    // parses the suboptions for profile-based approach
    private boolean parseProfileBased() {
        String[] values = Driver.getOptionValue("profitable-omp").split(":");
        if (!values[0].equals("2")) {
            return false;
        }
        option = PROFILE_BASED;
        if (values.length == 2) {
            for (String suboption : values[1].split(",")) {
                String msg = "Ignoring unknown suboption \'" + suboption + "\'";
                int subvalue;
                String[] subvalues = suboption.split("=");
                if (subvalues.length != 2) {
                    PrintTools.printlnStatus(0, pass_name, msg);
                    continue;
                }
                try {
                    subvalue = Integer.parseInt(subvalues[1]);
                } catch (NumberFormatException ex) {
                    PrintTools.printlnStatus(0, pass_name, msg);
                    continue;
                }
                if (subvalues[0].equals("grace")) {
                    profile_grace = subvalue;
                } else if (subvalues[0].equals("interval")) {
                    profile_interval = subvalue;
                } else if (subvalues[0].equals("repeat")) {
                    profile_repeat = subvalue;
                } else if (subvalues[0].equals("usmin")) {
                    profile_usmin = subvalue;
                } else {
                    PrintTools.printlnStatus(0, pass_name, msg);
                }
            }
        }
        return true;
    }

    private boolean parseHybrid() {
        String[] values = Driver.getOptionValue("profitable-omp").split(":");
        if (!values[0].equals("3")) {
            return false;
        }
        option = HYBRID;
        if (values.length == 2) {
            for (String suboption : values[1].split(",")) {
                String msg = "Ignoring unknown suboption \'" + suboption + "\'";
                int subvalue;
                String[] subvalues = suboption.split("=");
                if (subvalues.length != 2) {
                    PrintTools.printlnStatus(0, pass_name, msg);
                    continue;
                }
                try {
                    subvalue = Integer.parseInt(subvalues[1]);
                } catch (NumberFormatException ex) {
                    PrintTools.printlnStatus(0, pass_name, msg);
                    continue;
                }
                if (subvalues[0].equals("threshold")) {
                    model_threshold = subvalue;
                } else if (subvalues[0].equals("costwhile")) {
                    model_costwhile = subvalue;
                } else if (subvalues[0].equals("costfcall")) {
                    model_costfcall = subvalue;
                } else if (subvalues[0].equals("runtime")) {
                    model_runtime = subvalue;
                } else if (subvalues[0].equals("grace")) {
                    profile_grace = subvalue;
                } else if (subvalues[0].equals("interval")) {
                    profile_interval = subvalue;
                } else if (subvalues[0].equals("repeat")) {
                    profile_repeat = subvalue;
                } else if (subvalues[0].equals("usmin")) {
                    profile_usmin = subvalue;
                } else {
                    PrintTools.printlnStatus(0, pass_name, msg);
                }
            }
        }
        model_scostwhile = new IntegerLiteral(model_costwhile);
        model_scostfcall = new IntegerLiteral(model_costfcall);
        return true;
    }

    /**
    * Returns the name of this code generation pass.
    */
    public String getPassName() {
        return pass_name;
    }

    /**
    * Starts the code generation for profitable loop parallelization.
    */
    public void start() {
        DFIterator<Statement> iter =
                new DFIterator<Statement>(program, Statement.class);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(ExpressionStatement.class);
        int[] parcount = new int[] {0};
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            if (stmt.containsAnnotation(OmpAnnotation.class, "parallel")) {
                if (option == MODEL_BASED) {
                    addModelRuntime(stmt);
                } else if (option == PROFILE_BASED) {
                    addProfileRuntime(stmt, parcount);
                } else if (option == HYBRID) {
                    addModelRuntime(stmt);
                    addProfileRuntime(stmt, parcount);
                }
            }
        }
        if (option == PROFILE_BASED || option == HYBRID) {
            profile_numevents = parcount[0];
            finalizeProfileRuntime();
        }
    }

    // Codes for model-based approach
    /**
    * Computes symbolic workload with the predefined parameters.
    */
    private Expression countWorkLoad(Statement stmt) {
        Expression ret = model_szro;
        if (stmt instanceof ForLoop) {
            // TODO: handling of negative stride
            ForLoop floop = (ForLoop)stmt;
            Expression inc = LoopTools.getIncrementExpression(floop);
            Expression lb = LoopTools.getLowerBoundExpression(floop);
            Expression ub = LoopTools.getUpperBoundExpression(floop);
            Statement init = floop.getInitialStatement();
            Expression condition = floop.getCondition();
            Expression step = floop.getStep();
            Expression body_load = countWorkLoad(floop.getBody());
            // Load from the initial statement
            if (init != null) {
                ret = Symbolic.add(ret, model_sone);
                ret = Symbolic.add(ret, countFunctionCalls(init));
            }
            // Load from the condtion expression
            if (condition != null) {
                body_load = Symbolic.add(body_load, model_sone);
                body_load = Symbolic.add(body_load,
                                         countFunctionCalls(condition));
            }
            // Load from the step expression
            if (step != null) {
                body_load = Symbolic.add(body_load, model_sone);
                body_load = Symbolic.add(body_load, countFunctionCalls(step));
            }
            // Loop count is multiplied by the load from the loop body.
            if (inc != null && lb != null && ub != null) {
                Expression loop_count =
                        Symbolic.divide(Symbolic.add(
                                Symbolic.subtract(ub, lb), model_sone), inc);
                ret = Symbolic.add(ret,
                                   Symbolic.multiply(loop_count, body_load));
            // Use default loop count
            } else {
                ret = Symbolic.add(ret, Symbolic.multiply(
                        model_scostwhile, body_load));
            }
        // Other loops: condition and the body loads multiplied by a fixed
        // iteration
        } else if (stmt instanceof Loop) {
            Expression body_load = countWorkLoad(((Loop)stmt).getBody());
            Expression condition = ((Loop)stmt).getCondition();
            if (condition != null) {
                body_load = Symbolic.add(body_load, model_sone);
                body_load = Symbolic.add(body_load,
                                         countFunctionCalls(condition));
            }
            ret = Symbolic.add(ret, Symbolic.multiply(
                    model_scostwhile, body_load));
        // Compound statement: adds all workloads from the children
        } else if (stmt instanceof CompoundStatement) {
            List<Traversable> children = stmt.getChildren();
            int children_size = children.size();
            for (int i = 0; i < children_size; i++) {
                ret = Symbolic.add(ret,
                                   countWorkLoad((Statement)children.get(i)));
            }
        // IF/ELSE structure: adds all workloads for simplicity
        } else if (stmt instanceof IfStatement) {
            IfStatement if_stmt = (IfStatement)stmt;
            ret = Symbolic.add(ret, model_sone);
            ret = Symbolic.add(ret, countFunctionCalls(
                    if_stmt.getControlExpression()));
            ret = Symbolic.add(ret, countWorkLoad(if_stmt.getThenStatement()));
            if (if_stmt.getElseStatement() != null) {
                ret = Symbolic.add(ret,
                                   countWorkLoad(if_stmt.getElseStatement()));
            }
        // SWITCH structure: adds all workloads for simplicity
        } else if (stmt instanceof SwitchStatement) {
            ret = Symbolic.add(ret, model_sone);
            ret = Symbolic.add(ret, countWorkLoad(
                    ((SwitchStatement)stmt).getBody()));
        // Assumes every annotation does not contain any executable statements
        } else if (stmt instanceof AnnotationStatement) {
            ret = new IntegerLiteral(0);
        // Other cases
        } else {
            ret = Symbolic.add(ret, model_sone);
            ret = Symbolic.add(ret, countFunctionCalls(stmt));
        }
        return ret;
    }

    /** 
    * Returns workloads of the given traversable object by counting the number
    * of function calls within it.
    */
    private Expression countFunctionCalls(Traversable t) {
        int ret = 0;
        DFIterator<FunctionCall> iter =
                new DFIterator<FunctionCall>(t, FunctionCall.class);
        while (iter.hasNext()) {
            iter.next();
            ret++;
        }
        return Symbolic.multiply(model_scostfcall, new IntegerLiteral(ret));
    }

    /**
    * Computes and returns a valid workload expression for the given statement.
    * @param stmt the statement to be analyzed.
    * @return the expression for the workload, or null if there exist any
    *       invalid symbol within the expression.
    */
    private void addModelRuntime(Statement stmt) {
        OmpAnnotation omp = stmt.getAnnotation(OmpAnnotation.class, "parallel");
        if (omp != null) {
            Expression workload = countWorkLoad(stmt);
            // Cast literals to long integer
            DFIterator<IntegerLiteral> int_iter =
                new DFIterator<IntegerLiteral>(workload, IntegerLiteral.class);
            while (int_iter.hasNext()) {
                IntegerLiteral ilit = int_iter.next();
                ilit.swapWith(new IntegerLiteral(ilit.getValue(), "L"));
            }
            // Compile-time elimination
            if (workload instanceof IntegerLiteral) {
                if (((IntegerLiteral)workload).getValue() < model_threshold) {
                    String comment = "Disabled due to low profitability: ";
                    DFIterator<Statement> iter = new DFIterator<Statement>(
                            stmt, Statement.class);
                    while (iter.hasNext()) {
                        Statement s = iter.next();
                        List<OmpAnnotation> omps =
                                s.getAnnotations(OmpAnnotation.class);
                        for (OmpAnnotation oa : omps) {
                            oa.detach();
                            s.annotate(new CommentAnnotation(comment + oa));
                        }
                    }
                }
            // Run-time insertion
            } else if (option == MODEL_BASED && model_runtime != 0) {
                // Variable validity check
                Set<Expression> mods = DataFlowTools.getDefSet(stmt);
                SymbolTable parent =
                        IRTools.getAncestorOfType(stmt, SymbolTable.class);
                DFIterator<Identifier> iter =
                        new DFIterator<Identifier>(workload, Identifier.class);
                while (iter.hasNext()) {
                    Identifier id = iter.next();
                    if (SymbolTools.findSymbol(parent, id) == null ||
                        mods.contains(id)) {
                        workload = null;
                        break;
                    }
                }
                // Modified variable check
                if (workload != null) {
                    Expression condition = new BinaryExpression(
                            new IntegerLiteral(model_threshold),
                            BinaryOperator.COMPARE_LT,
                            workload);
                    omp.put("if", condition);
                }
            }
        }
    }

    private static final String[] HEADER = {
        "#ifndef HAS_SYS_TIME_H",
        "#include <sys/time.h>",
        "#endif",
        "typedef struct cetusrt_event {",
        "    int watch;",
        "    int profile;",
        "    int parallel;",
        "    int count;",
        "    int rem_interval;",
        "    int rem_profile;",
        "    int scale;",
        "    int prev;",
        "    struct timeval since;",
        "} cetusrt_event;",
        "inline void cetusrt_tic(cetusrt_event *);",
        "inline void cetusrt_toc(cetusrt_event *);",
        "extern cetusrt_event * CETUSRT_EVENT;"
    };

    private static final String[] LIBRARY = {
        "#include <stdlib.h>",
        "#ifndef HAS_SYS_TIME_H",
        "#include <sys/time.h>",
        "#endif",
        "",
        "typedef struct cetusrt_event {",
        "    int watch;",
        "    int profile;",
        "    int parallel;",
        "    int count;",
        "    int rem_interval;",
        "    int rem_profile;",
        "    int scale;",
        "    int prev;",
        "    struct timeval since;",
        "} cetusrt_event;",
        "",
        "inline void cetusrt_tic(cetusrt_event *evt) {",
        "    if (evt->profile) {",
        "        gettimeofday(&evt->since, 0);",
        "    }",
        "}",
        "",
        "inline void cetusrt_toc(cetusrt_event *evt) {",
        "    evt->count++;",
        "    if (evt->profile) {",
        "        struct timeval now;",
        "        long diff;",
        "        gettimeofday(&now, 0);",
        "        diff = 1000000*(now.tv_sec - evt->since.tv_sec) +",
        "               now.tv_usec - evt->since.tv_usec;",
        "        if (evt->parallel) {",
        "            if (evt->prev < CETUSRT_USMIN || diff < CETUSRT_USMIN) {",
        "                evt->parallel = 0;",
        "                evt->watch = 0;",
        "            } else {",
        "                evt->scale += (evt->prev > diff)? 1: -1;",
        "                evt->rem_profile--;",
        "                if (evt->scale > evt->rem_profile) {",
        "                    evt->parallel = 1;",
        "                    evt->watch = 0;",
        "                } else if (-evt->scale >= evt->rem_profile) {",
        "                    evt->parallel = 0;",
        "                    evt->watch = 0;",
        "                } else {",
        "                    evt->profile = 0;",
        "                    evt->rem_interval = CETUSRT_INTERVAL;",
        "                }",
        "            }",
        "        } else {",
        "            evt->prev = diff;",
        "            evt->parallel = 1;",
        "        }",
        "    } else if (evt->count==CETUSRT_GRACE || --evt->rem_interval==0) {",
        "        evt->profile = 1;",
        "        evt->parallel = 0;",
        "    }",
        "}",
        "",
        "cetusrt_event * cetusrt_initialize() {",
        "    int i;",
        "    cetusrt_event * ret = (cetusrt_event *)",
        "            malloc(CETUSRT_NUMEVENTS*sizeof(cetusrt_event));",
        "    for (i = 0; i < CETUSRT_NUMEVENTS; i++) {",
        "        cetusrt_event *evt = ret+i;",
        "        evt->watch = 1;",
        "        evt->profile = 0;",
        "        evt->parallel = 1;",
        "        evt->count = 0;",
        "        evt->rem_interval = 0;",
        "        evt->rem_profile = CETUSRT_REPEAT;",
        "        evt->scale = 0;",
        "        evt->prev = 0;",
        "    }",
        "    return ret;",
        "}",
        "",
        "cetusrt_event * CETUSRT_EVENT;"
    };

    /**
    * Inserts monitoring instrumentation around the given statement.
    */
    private void addProfileRuntime(Statement stmt, int[] id) {
        String sep = PrintTools.line_sep;
        OmpAnnotation omp = stmt.getAnnotation(OmpAnnotation.class, "parallel");
        if (omp != null) {
            int myid = id[0]++;
            String event_name = profile_name + "[" + myid + "]";
            // OpenMP IF clause
            omp.put("if", event_name + ".parallel");
            CompoundStatement parent =
                    IRTools.getAncestorOfType(stmt, CompoundStatement.class);
            // Tic
            Annotation code = new CodeAnnotation(
                    "#ifdef _OPENMP" + sep +
                    "if (" + event_name + ".watch) " +
                    "cetusrt_tic(" + profile_name + "+" + myid + ");" + sep +
                    "#endif");
            parent.addStatementBefore(stmt, new AnnotationStatement(code));
            // Toc
            code = new CodeAnnotation(
                    "#ifdef _OPENMP" + sep +
                    "if (" + event_name + ".watch) " +
                    "cetusrt_toc(" + profile_name + "+" + myid + ");" + sep +
                    "#endif");
            parent.addStatementAfter(stmt, new AnnotationStatement(code));
        }
    }

    /**
    * Finalizes the profiling runtime by inserting initializing code.
    */
    private void finalizeProfileRuntime() {
        String sep = PrintTools.line_sep;
        TranslationUnit main_tunit = null;
        Procedure main_proc = null;
        DFIterator<TranslationUnit> tu_iter =
                new DFIterator<TranslationUnit>(program, TranslationUnit.class);
        tu_iter.pruneOn(TranslationUnit.class);
        // Finds main procedure/file and exit statements while inserting headers
        while (tu_iter.hasNext()) {
            TranslationUnit tu = tu_iter.next();
            DFIterator<Traversable> iter = new DFIterator<Traversable>(tu);
            while (iter.hasNext()) {
                Traversable t = iter.next();
                if (t instanceof Procedure &&
                    ((Procedure)t).getSymbolName().equals("main")) {
                    main_proc = (Procedure)t;
                    main_tunit = tu;
                }
                /*
                else if (t instanceof FunctionCall &&
                    ((FunctionCall)t).getName().toString().equals("exit")) {
                    CompoundStatement parent = IRTools.getAncestorOfType(
                            t, CompoundStatement.class);
                    Statement exit_stmt = ((Expression)t).getStatement();
                    Annotation code = new CodeAnnotation(
                            "#ifdef _OPENMP" + sep +
                            "cetusrt_report(&" + mon_name + ");" + sep +
                            "#endif");
                    parent.addStatementBefore(
                            exit_stmt, new AnnotationStatement(code));
                }
                */
            }
            boolean has_stdlib = SymbolTools.findSymbol(tu, "malloc") != null;
            Declaration first_proc = null;
            List<Traversable> children = tu.getChildren();
            for (int i = children.size() - 1; i >= 0; i--) {
                Declaration decl = (Declaration)children.get(i);
                if (decl instanceof AnnotationDeclaration &&
                    decl.toString().contains("endinclude")) {
                    break;
                }
                if (decl instanceof Procedure) {
                    first_proc = decl;
                }
            }
            String code;
            if (tu == main_tunit) {
                code = genCode(LIBRARY);
            } else {
                code = genCode(HEADER);
            }
            /*
            if (has_stdio) {
                code = code.replace("#include <stdio.h>", "");
            }
            */
            if (has_stdlib) {
                code = code.replace("#include <stdlib.h>", "");
            }
            String parameters =
                    "#define CETUSRT_NUMEVENTS " + profile_numevents + sep +
                    "#define CETUSRT_GRACE " + profile_grace + sep +
                    "#define CETUSRT_INTERVAL " + profile_interval + sep +
                    "#define CETUSRT_REPEAT " + profile_repeat + sep +
                    "#define CETUSRT_USMIN " + profile_usmin + sep +
                    "#define CETUSRT_EVENT " + profile_name;
            Declaration decl = new AnnotationDeclaration(new CodeAnnotation(
                    "#ifdef _OPENMP" + sep +
                    parameters + sep +
                    code + sep +
                    "#endif"));
            if (first_proc != null) {
                tu.addDeclarationBefore(first_proc, decl);
            } else {
                tu.addDeclaration(decl);
            }
        }
        // Inserts library codes
        if (main_tunit == null) {
            Tools.exit("[ERROR] OMP runtime insertion needs a main function.");
        } else {
            // Finds first non-declaration statement
            Statement first_stmt = null;
            List<Traversable> children = main_proc.getBody().getChildren();
            for (int i = children.size() - 1; i >= 0; i--) {
                Statement stmt = (Statement)children.get(i);
                if (stmt instanceof DeclarationStatement) {
                    break;
                }
                first_stmt = stmt;
            }
            // Inserts initialization code
            String code =
                    "#ifdef _OPENMP" + sep +
                    "CETUSRT_EVENT = cetusrt_initialize();" + sep +
                    "#endif";
            Statement note = new AnnotationStatement(new CodeAnnotation(code));
            if (first_stmt == null) {
                main_proc.getBody().addStatement(note);
            } else {
                main_proc.getBody().addStatementBefore(first_stmt, note);
            }
            // Inserts exit code
            /*
            code =  "#ifdef _OPENMP" + sep +
                    "cetusrt_report(&" + mon_name + ");" + sep +
                    "#endif";
            DFIterator<ReturnStatement> ret_iter =
                    new DFIterator<ReturnStatement>(
                    main_proc, ReturnStatement.class);
            while (ret_iter.hasNext()) {
                ret_iter.next().annotateBefore(new CodeAnnotation(code));
            }
            Statement last_stmt = null;
            for (int i = children.size() -1; i >= 0; i--) {
                last_stmt = (Statement)children.get(i);
                if (!(last_stmt instanceof AnnotationStatement)) {
                    break;
                }
            }
            if (last_stmt != null && !(last_stmt instanceof ReturnStatement)) {
                last_stmt.annotateAfter(new CodeAnnotation(code));
            }
            */
        }
    }

    private static String genCode(String[] str) {
        String ret = "";
        for (String line : str) {
            ret += line + PrintTools.line_sep;
        }
        return ret;
    }

}

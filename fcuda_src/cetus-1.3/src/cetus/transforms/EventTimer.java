package cetus.transforms;

import java.util.*;
import cetus.hir.*;

/**
 * This pass instruments the program that contains
 * {@link cetus.hir.PragmaAnnotation.Event} annotations. The annotations may be
 * inserted either manually or automatically.
 */
public class EventTimer extends TransformPass {

    /** Line separator */
    private static final String NEWLINE = System.getProperty("line.separator");

    /** The variable name used for global profiling */
    private static final String prof_name = "cetus_prof";

    /** Pass name */
    private static final String pass_name = "[EventTimer]";

    /** Heading string for result printing */
    private static final String header = "CETUS_TIMING";

    /**
     * The contents of code to be prepended to each translation unit that does
     * not contain the program entry (main function). This code contains the
     * declaration of the library calls.
     */
    private static final String[] headercode = {
        "#ifdef " + header,
        "typedef struct cetusprofile cetusprofile;",
        "extern cetusprofile " + prof_name + ";",
        "void cetus_tic(cetusprofile *, int);",
        "void cetus_toc(cetusprofile *, int);",
        "#endif /* " + header + " */",
        ""
    };

    /**
     * The contents of code to be prepended to the translation unit that
     * contains the program entry. This code contains the definition of the
     * library calls.
     */
    private static final String[] libcode = {
        "#ifdef " + header,
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#ifndef HAS_SYS_TIME_H",
        "#include <sys/time.h>",
        "#endif",
        "/* structure for a timed event */",
        "struct timeval;",
        "struct cetusevent;",
        "typedef struct cetusevent {",
        "  long count;                 /* number of invocations        */",
        "  long etime;                 /* elapsed time in microseconds */",
        "  long child_count;           /* number of child invocations  */",
        "  long child_etime;           /* elapsed time of child event  */",
        "  long child_count_cumul;     /* cumulative child invocations */",
        "  char *name;                 /* name of the event            */",
        "  struct timeval since;       /* latest time stamp            */",
        "  struct cetusevent *parent;  /* reference to parent event    */",
        "} cetusevent;",
        "",
        "/* structure for global profiling */",
        "typedef struct cetusprofile {",
        "  int num_events;       /* number of timed events         */",
        "  long num_depths;      /* number of depths of all events */",
        "  long num_invocs;      /* number of invocated events     */",
        "  double overhead;      /* measured overhead per event    */",
        "  double overhead_in;   /* included overhead per event    */",
        "  cetusevent *current;  /* current ongoing event          */",
        "  cetusevent *event;    /* events to be profiled          */",
        "} cetusprofile;",
        "",
        "/* starts timing a single invocation of an event */",
        "void cetus_tic(cetusprofile *prof, int id)",
        "{",
        "  cetusevent *evt = prof->event+id;",
        "  evt->parent = prof->current;",
        "  prof->current = evt;",
        "  gettimeofday(&evt->since, 0);",
        "}",
        "",
        "/* finishes timing a single invocation of an event */",
        "void cetus_toc(cetusprofile *prof, int id)",
        "{",
        "  int i;",
        "  long diff;",
        "  struct timeval now;",
        "  cetusevent *ev, *evt = prof->event+id;",
        "",
        "  gettimeofday(&now, 0);",
        "  diff = 1000000 *",
        "      (now.tv_sec-evt->since.tv_sec)+now.tv_usec-evt->since.tv_usec;",
        "  /* updates count and elapsed time of the current event */",
        "  evt->count ++;",
        "  evt->etime += diff;",
        "  /* updates the child info of the parent event */",
        "  if (evt->parent) {",
        "    evt->parent->child_count ++;",
        "    evt->parent->child_etime += diff;",
        "  }",
        "  /* updates the number of cumulative child invocations */",
        "  for (i = 0, ev = evt->parent; ev; i++, ev = ev->parent)",
        "    ev->child_count_cumul ++;",
        "  /* updates global profile */",
        "  prof->num_depths += i;",
        "  prof->current = evt->parent;",
        "  evt->parent = 0;",
        "}",
        "",
        "/* initializes the data structure */",
        "void cetus_init_timers(cetusprofile *prof, char names[][32])",
        "{",
        "  int i;",
        "",
        "  /* initializes the event data */",
        "  prof->event = ",
        "    (cetusevent *)malloc(prof->num_events*sizeof(cetusevent));",
        "  for (i = 0; i < prof->num_events; i++) {",
        "    prof->event[i].count = 0;",
        "    prof->event[i].etime = 0;",
        "    prof->event[i].child_count = 0;",
        "    prof->event[i].child_etime = 0;",
        "    prof->event[i].child_count_cumul = 0;",
        "    prof->event[i].parent = 0;",
        "    if (names)",
        "      prof->event[i].name = names[i];",
        "  }",
        "}",
        "",
        "/* computes overheads from the profiling code */",
        "void cetus_tune_timer(cetusprofile *prof)",
        "{",
        "  int i, j, iter=1000000;",
        "  cetusprofile p = {0, 0, 0, 0.0, 0.0, 0, 0};",
        "  /* computes the average number of depths */",
        "  for (i = 0; i < prof->num_events; i++)",
        "    prof->num_invocs += prof->event[i].count;",
        "  p.num_events = prof->num_depths/prof->num_invocs + 2;",
        "  /* measures overheads from timing with simulated events */",
        "  cetus_init_timers(&p, 0);",
        "  for (i = 0; i < p.num_events-1; i++)",
        "    cetus_tic(&p, i);",
        "  for (j = 0; j < iter; j++) {",
        "    cetus_tic(&p, i);",
        "    cetus_toc(&p, i);",
        "  }",
        "  while (i-- > 0)",
        "    cetus_toc(&p, i);",
        "  prof->overhead = p.event[0].etime/(double)iter;",
        "  prof->overhead_in = p.event[p.num_events-1].etime/(double)iter;",
        "}",
        "",
        "/* prints the profiling result to the specified file stream */",
        "void cetus_print_timers(cetusprofile *prof, FILE * o)",
        "{",
        "  int i;",
        "  double sum_measured = 0, sum_adjusted = 0, sum_exclusive = 0;",
        "  double measured, adjusted, exclusive;",
        "  cetus_tune_timer(prof);",
        "  fprintf(o, \"\\n" + header + "%32s%12s%12s%12s%12s\\n\",",
        "  \"NAME\", \"INVOKED\", \"MEASURED\", \"ADJUSTED\", \"EXCLUSIVE\");",
        "  for (i = 0; i < prof->num_events; i++) {",
        "    cetusevent *evt = prof->event+i;",
        "    measured = 1.0e-6 * evt->etime;",
        "    adjusted = 1.0e-6 * (evt->etime",
        "        /* overhead from child's timing calls */",
        "        - prof->overhead * evt->child_count_cumul",
        "        /* included overhead for the event */",
        "        - prof->overhead_in * evt->count);",
        "    exclusive = 1.0e-6 * (evt->etime",
        "        /* included overhead for the event */",
        "        - prof->overhead_in*evt->count",
        "        /* overhead not included in the child's timing calls */",
        "        - (prof->overhead - prof->overhead_in) * evt->child_count",
        "        /* elapsed time and overhead included in the child */",
        "        - evt->child_etime);",
        "    if (i < prof->num_events-1) {",
        "      sum_measured += measured;",
        "      sum_adjusted += adjusted;",
        "      sum_exclusive += exclusive;",
        "    }",
        "    fprintf(o, \"" + header + "%32s%12.2e%12.2f%12.2f%12.2f\\n\",",
        "      evt->name, (double)evt->count, measured, adjusted, exclusive);",
        "  }",
        "  fprintf(o, \"" + header + "%32s%12.2e%12.2f%12.2f%12.2f\\n\",",
        "    \"EVENTSUM\", 0.0, sum_measured, sum_adjusted, sum_exclusive);",
        "  fprintf(o, \"" + header + "%32s%12.2e%12.2f%12.2f%12.2f\\n\",",
        "    \"COVERAGE\", 0.0, 100*(sum_measured/measured),",
        "                       100*(sum_adjusted/adjusted),",
        "                       100*(sum_exclusive/adjusted));",
        "}",
        "",
        "cetusprofile " + prof_name + " = {0, 0, 0, 0.0, 0.0, 0, 0};",
        "#endif /* " + header + " */"
    };

    /** The translation unit to be detected which contains the program entry */
    private TranslationUnit main_tunit;

    /** The main function to be detected */
    private Procedure main_proc;

    /** Number of events to be profiled */
    private int num_events;

    /** The list of event names to be collected */
    private List<String> event_names;

    /** The list of forced program exit points to be collected */
    private List<Statement> exit_stmts;

    /**
     * Constructs an event-timing instrumenter with the specified program.
     * @param prog the program to be instrumented.
     */
    public EventTimer(Program prog) {
        super(prog);
        main_tunit = null;
        main_proc = null;
        num_events = 0;
        event_names = new LinkedList<String>();
        exit_stmts = new LinkedList<Statement>();
        //disable_protection = true;
    }

    /** Returns the name of this pass */
    public String getPassName() {
        return pass_name;
    }

    /**
     * Performs instrumentation after analyzing the program to collect
     * information required to generate the profiling instrumentation.
     */
    public void start() {
        for (Traversable t : program.getChildren()) {
            transformTUnit((TranslationUnit)t);
        }
        if (main_tunit == null) {
            System.err.println("[WARNING] Program entry is missing");
        } else {
            // insert includes only if it is necessary
            String libcodes = genCode(libcode);
            if (SymbolTools.findSymbol(main_tunit, "printf") != null) {
                libcodes = libcodes.replace("#include <stdio.h>", "");
            }
            if (SymbolTools.findSymbol(main_tunit, "malloc") != null) {
                libcodes = libcodes.replace("#include <stdlib.h>", "");
            }
            Declaration libcode_decl =
                    new AnnotationDeclaration(new CodeAnnotation(libcodes));
            Declaration first_proc = getFirstProcedure(main_tunit);
            if (first_proc == null) { // should not happen in general
                main_tunit.addDeclaration(libcode_decl);
            } else {
                main_tunit.addDeclarationBefore(first_proc, libcode_decl);
            }
            // creates/inserts initialization code
            List<String> codes = new LinkedList<String>();
            codes.add("#ifdef " + header);
            codes.add("char " + prof_name + "_names[][32] = {");
            for (String name : event_names) {
                codes.add("    \"" + name + "\", ");
            }
            codes.add("    \"PROGRAM\"");
            codes.add("};");
            codes.add(prof_name + ".num_events = " + (num_events + 1) + ";");
            codes.add("cetus_init_timers(&" + prof_name + ", " + prof_name +
                    "_names);");
            codes.add("cetus_tic(&" + prof_name + ", " + num_events + ");");
            codes.add("#endif /* " + header + " */");
            Statement first_stmt =
                    getFirstStatementAfterDeclaration(main_proc.getBody());
            AnnotationStatement note_stmt =
                    new AnnotationStatement(new CodeAnnotation(genCode(codes)));
            if (first_stmt == null) {
                main_proc.getBody().addStatement(note_stmt);
            } else {
                main_proc.getBody().addStatementBefore(first_stmt, note_stmt);
            }
            codes.clear();
            // creates/inserts exiting code
            codes.add("#ifdef " + header);
            codes.add("cetus_toc(&" + prof_name + ", " + num_events + ");");
            codes.add("cetus_print_timers(&" + prof_name + ", stderr);");
            codes.add("#endif");
            // program exit
            for (Statement exit : exit_stmts) {
                exit.annotateBefore(new CodeAnnotation(genCode(codes)));
            }
            // return statement
            DFIterator<ReturnStatement> iter = new DFIterator<ReturnStatement>(
                    main_proc.getBody(), ReturnStatement.class);
            while (iter.hasNext()) {
                iter.next().annotateBefore(new CodeAnnotation(genCode(codes)));
            }
            // last non-return statement
            List<Traversable> children = main_proc.getBody().getChildren();
            Statement last_stmt = (Statement)children.get(children.size() - 1);
            if (!(last_stmt instanceof ReturnStatement)) {
                last_stmt.annotateAfter(new CodeAnnotation(genCode(codes)));
            }
        }
    }

    /**
     * Inserts timing library calls at the location specified by event-type
     * annotation.
     * @param tunit the translation unit to be transformed.
     */
    public void transformTUnit(TranslationUnit tunit) {
        DFIterator<Traversable> iter = new DFIterator<Traversable>(tunit);
        iter.pruneOn(VariableDeclaration.class);
        while (iter.hasNext()) {
            Traversable t = iter.next();
            if (t instanceof Procedure
                    && ((Procedure)t).getSymbolName().equals("main")) {
                main_proc = (Procedure) t;
                main_tunit = tunit;
            } else if (t instanceof Statement) {
                Statement stmt = (Statement)t;
                PragmaAnnotation.Event event =
                        stmt.getAnnotation(PragmaAnnotation.Event.class,"name");
                if (event != null) {
                    String fcall = "";
                    String name = event.getName(), command = event.getCommand();
                    if (command.equals("start")) {
                        fcall = "cetus_tic(&" + prof_name + ", " +
                                (num_events++) + ");";
                        event_names.add(name);
                    } else if (command.equals("stop")) {
                        int event_num = event_names.indexOf(name);
                        fcall = "cetus_toc(&" + prof_name + ", " +
                                event_num + ");";
                    } else {
                        throw new InternalError(pass_name +
                                " Unknown event pragma");
                    }
                    stmt.annotate(new CodeAnnotation(
                            "#ifdef " + header + NEWLINE
                            + fcall + NEWLINE
                            + "#endif"));
                }
            } else if (t instanceof FunctionCall
                    && ((FunctionCall)t).getName().toString().equals("exit")) {
                exit_stmts.add(((Expression)t).getStatement());
            }
        }
        if (tunit != main_tunit) {
            tunit.addDeclarationFirst(new AnnotationDeclaration(
                    new CodeAnnotation(genCode(headercode))));
        }
    }

    /** Returns the first statement after declaration part */
    private static Statement
            getFirstStatementAfterDeclaration(CompoundStatement cstmt) {
        Statement ret = null;
        List<Traversable> children = cstmt.getChildren();
        for (int i = children.size() - 1; i >= 0; i--) {
            Statement stmt = (Statement)children.get(i);
            if (stmt instanceof DeclarationStatement) {
                break;
            }
            ret = stmt;
        }
        return ret;
    }

    /** Returns the procedure that lexically appears first in the user code */
    private Declaration getFirstProcedure(TranslationUnit tu) {
        Declaration ret = null;
        List<Traversable> children = tu.getChildren();
        for (int i = children.size() - 1; i >= 0; i--) {
            Declaration decl = (Declaration)children.get(i);
            if (decl instanceof AnnotationDeclaration &&
                decl.toString().contains("endinclude")) {
                break;
            }
            if (decl instanceof Procedure) {
                ret = decl;
            }
        }
        return ret;
    }

    /** Generates code with the given list of lines */
    private String genCode(List<String> code) {
        String ret = "";
        for (String line : code) {
            ret += line + NEWLINE;
        }
        return ret;
    }

    /** Generates code with the given list of lines */
    private String genCode(String[] code) {
        String ret = "";
        for (String line : code) {
            ret += line + NEWLINE;
        }
        return ret;
    }

}

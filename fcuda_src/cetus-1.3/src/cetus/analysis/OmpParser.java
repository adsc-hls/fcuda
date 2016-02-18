package cetus.analysis;

import cetus.hir.PrintTools;
import cetus.hir.Tools;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * an OpenMP directive parser
 */
public class OmpParser {

    private static String [] token_array;
    private static int token_index;
    private static HashMap omp_map;
    private int debug_level;

    public OmpParser() {
    }

    private static String get_token() {
        return token_array[token_index++];
    }

    // consume one token
    private static void eat() {
        token_index++;
    }

    // match a token with the given string
    private static boolean match(String istr) {
        boolean answer = check(istr);
        if (answer == false) {
            System.out.println("OmpParser Syntax Error");
            System.out.println(display_tokens());
        }
        token_index++;
        return answer;
    }

    // match a token with the given string, but do not consume a token
    private static boolean check(String istr) {
        if (end_of_token()) { 
            return false;
        }
        return (token_array[token_index].compareTo(istr) == 0) ? true : false;
    }    

    private static String display_tokens() {
        StringBuilder str = new StringBuilder(160);
        for (int i = 0; i < token_array.length; i++) {
            str.append("token_array[").append(i).append("] = ");
            str.append(token_array[i]).append(PrintTools.line_sep);
        }
        return str.toString();
    }

    private static boolean end_of_token() {
        return (token_index >= token_array.length) ? true : false;
    }

    // returns TRUE, if the omp pragma will be attached to the following
    // non-pragma statement. Or, it returns false, if the pragma is a
    // standalone pragma
    public static boolean
            parse_omp_pragma(HashMap input_map, String [] str_array) {
        omp_map = input_map;
        token_array = str_array;
        token_index = 3;    // "#", "pragma", "omp" have already been matched
        PrintTools.println(display_tokens(), 9);
        String construct = "omp_" + get_token();
        switch (omp_pragma.valueOf(construct)) {
            case omp_parallel       : parse_omp_parallel();     return true;
            case omp_for            : parse_omp_for();          return true;
            case omp_sections       : parse_omp_sections();     return true;
            case omp_section        : parse_omp_section();      return true;
            case omp_single         : parse_omp_single();       return true;
            case omp_task           : parse_omp_task();         return true;
            case omp_master         : parse_omp_master();       return true;
            case omp_critical       : parse_omp_critical();     return true;
            case omp_barrier        : parse_omp_barrier();      return false;
            case omp_taskwait       : parse_omp_taskwait();     return false;
            case omp_atomic         : parse_omp_atomic();       return true;
            case omp_flush          : parse_omp_flush();        return false;
            case omp_ordered        : parse_omp_ordered();      return true;
            case omp_threadprivate  : parse_omp_threadprivate();return false;
            default                 : OmpParserError("Not Supported Construct");
        }
        return true;        // meaningless return because it is unreachable
    }

    /** ---------------------------------------------------------------
      *        2.4 parallel Construct
      *
      *        #pragma omp parallel [clause[[,] clause]...] new-line
      *            structured-block
      *
      *        where clause is one of the following
      *            if(scalar-expression)
      *            num_threads(integer-expression)
      *            default(shared|none)
      *            private(list)
      *            firstprivate(list)
      *            shared(list)
      *            copyin(list)
      *            reduction(operator:list)
      * --------------------------------------------------------------- */
    private static void parse_omp_parallel() {
        PrintTools.println("OmpParser is parsing [parallel] clause", 2);
        addToMap("parallel", "true");
        if (check("for")) {
            eat();
            parse_omp_parallel_for();
        } else if (check("sections")) {
            eat();
            parse_omp_parallel_sections();
        } else {
            while (end_of_token() == false) {
                String clause = "token_" + get_token();
                PrintTools.println("clause=" + clause, 2);
                switch (omp_clause.valueOf(clause)) {
                    case token_if           : parse_omp_if();           break;
                    case token_num_threads  : parse_omp_num_threads();  break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_copyin       : parse_omp_copyin();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    default                 :
                        OmpParserError("NoSuchParallelConstruct : " + clause);
                }
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.5 Worksharing Constructs
      *        OpenMP defines the following worksharing constructs
      *        - loop, sections, single, workshare(FORTRAN only) construct
      * --------------------------------------------------------------- */
    /** ---------------------------------------------------------------
      *        2.5.1 Loop Construct
      *
      *        #pragma omp for [clause[[,] clause]...] new-line
      *            for-loops
      *        where clause is one of the following
      *            private(list)
      *            firstprivate(list)
      *            lastprivate(list)
      *            reduction(operator:list)
      *            schedule(kind[, chunk_size])
      *            collapse(n)
      *            ordered
      *            nowait
      * --------------------------------------------------------------- */
    private static void parse_omp_for() {
        PrintTools.println("OmpParser is parsing [for] clause", 2);
        addToMap("for", "true");
        while (end_of_token() == false) {
            String clause = "token_" + get_token();
            switch (omp_clause.valueOf(clause)) {
                case token_private      : parse_omp_private();      break;
                case token_shared       : parse_omp_shared();       break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_schedule     : parse_omp_schedule();     break;
                case token_collapse     : parse_omp_collapse();     break;
                case token_ordered      : parse_omp_ordered();      break;
                case token_nowait       : parse_omp_nowait();       break;
                default : OmpParserError("NoSuchParallelConstruct");
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.5.2 sections Construct
      *
      *        #pragma omp sections [clause[[,] clause]...] new-line
      *        {
      *            [#pragma omp section new-line]
      *                structured-block
      *            [#pragma omp section new-line
      *                structured-block]
      *        }
      *        where clause is one of the following
      *            private(list)
      *            firstprivate(list)
      *            lastprivate(list)
      *            reduction(operator:list)
      *            nowait
      * --------------------------------------------------------------- */
    private static void parse_omp_sections() {
        PrintTools.println("OmpParser is parsing [sections] clause", 2);
        addToMap("sections", "true");
        while (end_of_token() == false) {
            String clause = "token_" + get_token();
            switch (omp_clause.valueOf(clause)) {
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_nowait       : parse_omp_nowait();       break;
                default : OmpParserError("NoSuchParallelConstruct");
            }
        }
    }

    private static void parse_omp_section() {
        PrintTools.println("OmpParser is parsing [section] clause", 2);
        addToMap("section", "true");
    }

    /** ---------------------------------------------------------------
      *        2.5.3 single Construct
      *
      *        #pragma omp single [clause[[,] clause]...] new-line
      *                structured-block
      *
      *        where clause is one of the following
      *            private(list)
      *            firstprivate(list)
      *            copyprivate(list)
      *            nowait
      * --------------------------------------------------------------- */
    private static void parse_omp_single() {
        PrintTools.println("OmpParser is parsing [single] clause", 2);
        addToMap("single", "true");
        while (end_of_token() == false) {
            String clause = "token_" + get_token();
            switch (omp_clause.valueOf(clause)) {
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_copyprivate  : parse_omp_copyprivate();  break;
                case token_nowait       : parse_omp_nowait();       break;
                default : OmpParserError("NoSuchParallelConstruct");
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.6 Combined Parallel Worksharing Constructs
      *
      *        2.6.1 parallel loop Construct
      *
      *        #pragma omp parallel for [clause[[,] clause]...] new-line
      *                for-loop
      *        
      *        where clause can be any of the clauses accepted by the parallel
      *        or for directives, except the nowait clause, with identical
      *        meanings and restrictions
      * --------------------------------------------------------------- */
    private static void parse_omp_parallel_for() {
        PrintTools.println("OmpParser is parsing [parallel for] clause", 2);
        addToMap("for", "true");
        while (end_of_token() == false) {
            String clause = "token_" + get_token();
            switch (omp_clause.valueOf(clause)) {
                case token_if           : parse_omp_if();           break;
                case token_num_threads  : parse_omp_num_threads();  break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_copyin       : parse_omp_copyin();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_schedule     : parse_omp_schedule();     break;
                case token_collapse     : parse_omp_collapse();     break;
                case token_ordered      : parse_omp_ordered();      break;
                default : OmpParserError("NoSuchParallelConstruct");
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.6.2 parallel sections Construct
      *
      *        #pragma omp sections [clause[[,] clause]...] new-line
      *        {
      *            [#pragma omp section new-line]
      *                structured-block
      *            [#pragma omp section new-line
      *                structured-block]
      *        }
      *        
      *        where clause can be any of the clauses accepted by the parallel
      *        or sections directives, except the nowait clause, with identical
      *        meanings and restrictions
      * --------------------------------------------------------------- */
    private static void parse_omp_parallel_sections() {
        PrintTools.println("OmpParser is parsing [parallel sections] clause",2);
        while (end_of_token() == false) {
            String clause = "token_" + get_token();
            addToMap("sections", "true");
            switch (omp_clause.valueOf(clause)) {
                case token_if           : parse_omp_if();           break;
                case token_num_threads  : parse_omp_num_threads();  break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_copyin       : parse_omp_copyin();       break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                default : OmpParserError("NoSuchParallelConstruct");
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.7 task Construct
      *
      *        #pragma omp task [clause[[,] clause]...] new-line
      *            structured-block
      *
      *        where clause is one of the following
      *            if(scalar-expression)
      *            untied
      *            default(shared|none)
      *            private(list)
      *            firstprivate(list)
      *            shared(list)
      * --------------------------------------------------------------- */
    private static void parse_omp_task() {
        PrintTools.println("OmpParser is parsing [task] clause", 2);
        while (end_of_token() == false) {
            String clause = "token_" + get_token();
            addToMap("task", "true");
            switch (omp_clause.valueOf(clause)) {
                case token_if           : parse_omp_if();           break;
                case token_untied       : parse_omp_untied();       break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                default : OmpParserError("NoSuchParallelConstruct");
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.8 Master and Synchronization Construct
      *
      *        -    master/critical/barrier/taskwait/atomic/flush/ordered
      *
      *        2.8.1 master Construct
      *
      *        #pragma omp master new-line
      *            structured-block
      *
      * --------------------------------------------------------------- */
    private static void parse_omp_master() {
        PrintTools.println("OmpParser is parsing [master] clause", 2);
        addToMap("master", "true");
    }

    private static void parse_omp_critical() {
        PrintTools.println("OmpParser is parsing [critical] clause", 2);
        String name = null;
        if (end_of_token() == false) {
            match("(");
            name = new String(get_token());
            match(")");
        }
        addToMap("critical", name);
    }

    private static void parse_omp_barrier() {
        PrintTools.println("OmpParser is parsing [barrier] clause", 2);
        addToMap("barrier", "true");
    }

    private static void parse_omp_taskwait() {
        PrintTools.println("OmpParser is parsing [taskwait] clause", 2);
        addToMap("taskwait", "true");
    }

    private static void parse_omp_atomic() {
        PrintTools.println("OmpParser is parsing [atomic] clause", 2);
        addToMap("atomic", "true");
    }

    private static void parse_omp_flush() {
        PrintTools.println("OmpParser is parsing [flush] clause", 2);
        Set set = new HashSet<String>();
        if (end_of_token() == false) {
            match("(");
            parse_commaSeparatedList(set);
            match(")");
        }
        addToMap("flush", set);
    }

    private static void parse_omp_ordered() {
        PrintTools.println("OmpParser is parsing [ordered] clause", 2);
        addToMap("ordered", "true");
    }

    /** ---------------------------------------------------------------
      *        2.9 Data Environment
      *
      *        2.9.1 read the specification
      *
      *        2.9.2 threadprivate Directive
      *
      * --------------------------------------------------------------- */
    /**
      * threadprivate(x) needs special handling: it should be a global
      * information that every parallel region has to be annotated as private(x)
      */
    private static void parse_omp_threadprivate() {
        PrintTools.println("OmpParser is parsing [threadprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("threadprivate", set);
    }
    
    /** ---------------------------------------------------------------
      *
      *        A collection of parser routines for OpenMP clauses
      *
      * --------------------------------------------------------------- */
    /**
      * This function parses a list of strings between a parenthesis, for
      * example, (scalar-expression) or (integer-expression).
      */
    private static String parse_ParenEnclosedExpr() {
        String str = null;
        int paren_depth = 1;
        match("(");
        while (true) {
            if (check("(")) {
                paren_depth++;
            }
            if (check(")")) {
                if (--paren_depth==0) {
                    break;
                }
            }
            if (str == null) {
                str = new String(get_token());
            } else {
                str.concat((" " + get_token()));
            }
        }
        match(")");
        return str;
    }

    // it is assumed that a (scalar-expression) is of the form (size < N)
    private static void parse_omp_if() {
        PrintTools.println("OmpParser is parsing [if] clause", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("if", str);
    }

    // it is assumed that a (integer-expression) is of the form (4)
    private static void parse_omp_num_threads() {
        PrintTools.println("OmpParser is parsing [omp_num_threads]", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("num_threads", str);    
    }

    /**
      * schedule(kind[, chunk_size])
      */
    private static void parse_omp_schedule() {
        PrintTools.println("OmpParser is parsing [schedule] clause", 2);
        String str = null;
        match("(");
        // schedule(static, chunk_size)
        // schedule(dynamic, chunk_size)
        // schedule(guided, chunk_size)
        if (check("static") || check("dynamic") || check("guided")) { 
            str = new String(get_token());
            if (check(",")) {
                match(",");
                eat();        // consume "chunk_size"    
            }
        // schedule(auto), schedule(runtime)
        } else if (check("auto") || check("runtime")) {
            str = new String(get_token());
        } else {
            OmpParserError("No such scheduling kind");
        }
        match(")");
        addToMap("schedule", str);
    }

    private static void parse_omp_collapse() {
        PrintTools.println("OmpParser is parsing [collapse] clause", 2);
        match("(");
        String int_str = new String(get_token());
        match(")");
        addToMap("collapse", int_str);    
    }

    private static void parse_omp_nowait() { 
        PrintTools.println("OmpParser is parsing [nowait] clause", 2);
        addToMap("nowait", "true"); 
    }

    private static void parse_omp_untied() { 
        PrintTools.println("OmpParser is parsing [untied] clause", 2);
        addToMap("untied", "true"); 
    }

    private static void parse_omp_default() {
        PrintTools.println("OmpParser is parsing [default] clause", 2);
        match("(");
        if (check("shared") || check("none")) {
            addToMap("default", new String(get_token()));
        } else {
            OmpParserError("NoSuchParallelDefaultCluase");
        }
        match(")");
    }

    private static void parse_omp_private() {
        PrintTools.println("OmpParser is parsing [private] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("private", set);
    }

    private static void parse_omp_firstprivate() {
        PrintTools.println("OmpParser is parsing [firstprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("firstprivate", set);
    }

    private static void parse_omp_lastprivate() {
        PrintTools.println("OmpParser is parsing [lastprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("lastprivate", set);
    }

    private static void parse_omp_copyprivate() {
        PrintTools.println("OmpParser is parsing [copyprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("copyprivate", set);
    }

    private static void parse_omp_shared() {
        PrintTools.println("OmpParser is parsing [shared] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("shared", set);
    }

    private static void parse_omp_copyin() {
        PrintTools.println("OmpParser is parsing [copyin] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("copyin", set);
    }

    // reduction(oprator:list)
    @SuppressWarnings("unchecked")
    private static void parse_omp_reduction() {
        PrintTools.println("OmpParser is parsing [reduction] clause", 2);
        HashMap reduction_map = null;
        Set set = null;
        String op = null;
        match("(");
        // Discover the kind of reduction operator (+, etc)
        if (check("+") || check("*") || check("-") || check("&") ||
            check("|") || check("^") || check("&&") || check("||")) {
            op = get_token();
            PrintTools.println("reduction op:" + op, 2);
        } else {
            OmpParserError("Undefined reduction operator");
        }

        // check if there is already a reduction annotation with the same
        // operator in the set
        for (String ikey : (Set<String>)(omp_map.keySet())) {
            if (ikey.compareTo("reduction") == 0) {
                reduction_map = (HashMap)(omp_map.get(ikey));
                set = (Set)(reduction_map.get(op));
                break;
            }
        }
        if (reduction_map == null) {
            reduction_map = new HashMap(4);
        } 
        if (match(":") == false) {
            OmpParserError(
                    "colon expected before a list of reduction variables");
        }
        // When reduction_map is not null, set can be null for the given
        // reduction operator
        if (set == null) {
            set = new HashSet<String>();
        }
        parse_commaSeparatedList(set);
        match(")");
        reduction_map.put(op, set);
        addToMap("reduction", reduction_map);
    }

    /**
      * This function reads a list of comma-separated variables
      * It checks the right parenthesis to end the parsing, but does not
      * consume it.
      */
    @SuppressWarnings("unchecked")
    private static void parse_commaSeparatedList(Set set) {
        for (;;) {
            set.add(get_token());
            if (check(")")) {
                break;
            } else if (match(",") == false) {
                OmpParserError("comma expected in comma separated list");
            }
        }
    }

    private static void notSupportedWarning(String text) {
        System.out.println("Not Supported OpenMP feature: " + text); 
    }

    private static void OmpParserError(String text) {
        System.out.println("OpenMP Parser Syntax Error: " + text);
        System.out.println(display_tokens());
    }

    @SuppressWarnings("unchecked")
    private static void addToMap(String key, String new_str) {
        if (omp_map.keySet().contains(key)) {
            Tools.exit("[Warning] OMP Parser detected duplicate pragma: "+key);
        } else {
            omp_map.put(key, new_str);
        }
    }

    // When a key already exists in the map
    // from page 31, 2.9.3 Data-Sharing Attribute Clauses
    // With the exception of the default clause, clauses may be repeated as
    // needed. A list item that specifies a given variable may not appear in
    // more than one clause on the same directive, except that a variable may
    // be specified in both firstprivate and lastprivate clauses.
    @SuppressWarnings("unchecked")
    private static void addToMap(String key, Set new_set) {
        if (omp_map.keySet().contains(key)) {
            Set set = (Set)omp_map.get(key);
            set.addAll(new_set);
        } else {
            omp_map.put(key, new_set);
        }
    }

    // reduction clause can be repeated as needed
    @SuppressWarnings("unchecked")
    private static void addToMap(String key, Map new_map) {
        if (omp_map.keySet().contains(key)) {
            Map orig_map = (Map)omp_map.get(key);
            for (String new_str : (Set<String>)new_map.keySet()) {
                Set new_set = (Set)new_map.get(new_str);
                if (orig_map.keySet().contains(new_str)) {
                    Set orig_set = (Set)orig_map.get(new_str);
                    orig_set.addAll(new_set);
                } else {
                    orig_map.put(new_str, new_set);
                }
            }
        } else {
            omp_map.put(key, new_map);
        }
    }

    public static enum omp_pragma {
        omp_parallel, 
        omp_for, 
        omp_sections, 
        omp_section, 
        omp_single, 
        omp_task, 
        omp_master, 
        omp_critical, 
        omp_barrier, 
        omp_taskwait,
        omp_atomic, 
        omp_flush, 
        omp_ordered,
        omp_threadprivate
    }

    public static enum omp_clause {
        token_if,
        token_num_threads,
        token_default,
        token_private,
        token_firstprivate,
        token_lastprivate,
        token_shared,
        token_copyprivate,
        token_copyin,
        token_schedule,
        token_nowait,
        token_ordered,
        token_untied,
        token_collapse,
        token_reduction
    }
}

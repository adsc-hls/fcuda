package cetus.analysis;

import cetus.hir.PrintTools;
import cetus.hir.Tools;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Parses string text within a Cetus annotation of type #pragma cetus ...
 * The parsed version is indexed by possible keys and the output is stored in
 * a map
 */
public class CetusAnnotationParser {

    private static String [] token_array;

    private static int token_index;

    private static HashMap cetus_map;

    public CetusAnnotationParser() {
    }

    private static String get_token() {
        return token_array[token_index++];
    }

    private static String get_prev_token() {
        return token_array[token_index-1];
    }
    
    /* Consume the current token and move to the next */
    private static void eat() {
        token_index++;
    }

    /* Match a token with the input istr, then move to the next token */
    private static boolean match(String istr) {
        boolean answer = check(istr);
        if (answer == false) {
            System.out.println("CetusAnnotationParser Syntax Error");
            System.out.println(display_tokens());
        }
        token_index++;
        return answer;
    }

    /* Match a token with the given string, but don't comsume the token */
    private static boolean check(String istr) {
        if (end_of_token()) { 
            return false;
        }
        return (token_array[token_index].compareTo(istr) == 0);
    }    

    /* Check what the previous token was */
    private static boolean check_prev_token(String istr) {
        if (start_of_token()) {
            return false;
        }
        return (token_array[token_index-1].compareTo(istr) == 0);
    } 
    
    private static String display_tokens() {
        StringBuilder str = new StringBuilder(160);
        for (int i = 0; i < token_array.length; i++) {
            str.append("token_array[" + i + "] = " + token_array[i] + "\n");
        }
        return str.toString();
    }

    private static boolean end_of_token() {
        return (token_index >= token_array.length);
    }

    private static boolean start_of_token() {
        return (token_index == 0);
    }
    
    /*
     * Parses pragmas that can be used with the "cetus" keyword
     * Currently supports parallelization annotations such as:
     * #pragma cetus private(t,s)
     * #pragma cetus reduction(+:s)
     * #pragma cetus parallel
     * #pragma cetus parallel private(t,s) reduction(+:s) 
     * @param input_map
     * @param str_array
     * @return
     */
    public static boolean parse_pragma(HashMap input_map, String [] str_array) {
        boolean attachable_pragma = false;
        cetus_map = input_map;
        token_array = str_array;
        /* Start from token 3 as #, pragma, cetus have already been matched */
        token_index = 3;
        PrintTools.println(display_tokens(), 9);
        while (!end_of_token()) {
            String construct = "cetus_" + get_token();
            switch (cetus_pragma.valueOf(construct)) {
                case cetus_parallel:
                    parse_cetus_parallel();
                    attachable_pragma = true;
                    break;
                case cetus_use:
                    parse_cetus_use();
                    attachable_pragma = true;
                    break;
                case cetus_def:
                    parse_cetus_def();
                    attachable_pragma = true;
                    break;
                case cetus_private:
                    parse_cetus_private();
                    attachable_pragma = true;
                    break;
                case cetus_firstprivate:
                    parse_cetus_firstprivate();
                    attachable_pragma = true;
                    break;
                case cetus_lastprivate:
                    parse_cetus_lastprivate();
                    attachable_pragma = true;
                    break;
                case cetus_reduction:
                    parse_cetus_reduction();
                    attachable_pragma = true;
                    break;
                default : CetusAnnotationParserError("Not Supported Construct");
            }
        }
        return attachable_pragma;
    }

    private static void parse_cetus_parallel() {
        PrintTools.println(
                "CetusAnnotationParser is parsing [parallel] clause", 2);
        addToMap("parallel", "true");
        while (end_of_token() == false) {
            String clause = "token_" + get_token();
            PrintTools.println("clause=" + clause, 2);
            switch (cetus_clause.valueOf(clause)) {
            case token_private      : parse_cetus_private(); break;
            case token_firstprivate : parse_cetus_firstprivate(); break;
            case token_lastprivate  : parse_cetus_lastprivate(); break;
            case token_reduction    : parse_cetus_reduction(); break;
            default                 : CetusAnnotationParserError(
                                      "NoSuchParallelConstruct : " + clause);
            }
        }
    }

    /** ---------------------------------------------------------------
     *
     *        A collection of parser routines for OpenMP clauses
     *
     * --------------------------------------------------------------- */
    private static void parse_cetus_use() {
        PrintTools.println("CetusAnnotationParser is parsing [use] clause", 2);
        match("(");
        Set<String> set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("use", set);
    }

    private static void parse_cetus_def() {
        PrintTools.println("CetusAnnotationParser is parsing [def] clause", 2);
        match("(");
        Set<String> set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("def", set);
    }

    private static void parse_cetus_private() {
        PrintTools.println(
                "CetusAnnotationParser is parsing [private] clause", 2);
        match("(");
        Set<String> set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("private", set);
    }

    private static void parse_cetus_firstprivate() {
        PrintTools.println(
                "CetusAnnotationParser is parsing [firstprivate] clause", 2);
        match("(");
        Set<String> set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("firstprivate", set);
    }

    private static void parse_cetus_lastprivate() {
        PrintTools.println(
                "CetusAnnotationParser is parsing [lastprivate] clause", 2);
        match("(");
        Set<String> set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("lastprivate", set);
    }

    // reduction(oprator:list)
    @SuppressWarnings("unchecked")
    private static void parse_cetus_reduction() {
        PrintTools.println(
                "CetusAnnotationParser is parsing [reduction] clause", 2);
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
            CetusAnnotationParserError("Undefined reduction operator");
        }
        // check if there is already a reduction annotation with the same
        // operator in the set
        for (String ikey : (Set<String>)(cetus_map.keySet())) {
            if (ikey.compareTo("reduction") == 0) {
                reduction_map = (HashMap)(cetus_map.get(ikey));
                set = (Set)(reduction_map.get(op));
                break;
            }
        }
        if (reduction_map == null) {
            reduction_map = new HashMap(4);
        } 
        if (!match(":")) {
            CetusAnnotationParserError(
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
     * This function reads a list of comma-separated variables. It checks the
     * right parenthesis to end the parsing, but does not consume it
     */
    @SuppressWarnings("unchecked")
    private static void parse_commaSeparatedList(Set set) {
        String str = new String();
        for (;;) {
            // We have a nested paren string, parse it as one whole
            // If prev token wasn't a comma, we need to attach the previous str
            // to the string we're about to parse as one whole
            if (check ("(")) {
                str = str.concat(parse_ParenEnclosedExpr());
            }
            if (check(")")) {
                if (str.length() != 0) {
                    set.add(str);
                }
                break;
            }
            if (check(",")) {
                set.add(str);
                match(",");
                str = new String();
            } else if (check("(")) {
                continue;
            } else {
                str = str.concat(get_token());
            }
        }
    }

    /**
     * This function parses a list of strings between a parenthesis, for
     * example, expressions enclosed inside parenthesis
     */
    private static String parse_ParenEnclosedExpr() {
        String enclosed_str = new String();
        int paren_depth = 1;
        match("(");
        enclosed_str = "(";
        while (true) {
            if (check("(")) {
                match("(");
                enclosed_str = enclosed_str.concat("(");
                paren_depth++;
            } else if (check(")")) {
                if (paren_depth-- == 1) {
                    break;
                }
                match(")");
                enclosed_str = enclosed_str.concat(")");
            } else {
                enclosed_str = enclosed_str.concat(get_token());
            }
        }
        match(")");
        enclosed_str = enclosed_str.concat(")");
        return enclosed_str;
    }

    private static void CetusAnnotationParserError(String text) {
        System.out.println("Cetus Annotation Parser Syntax Error: " + text);
        System.out.println(display_tokens());
    }

    @SuppressWarnings("unchecked")
    private static void addToMap(String key, String new_str) {
        if (cetus_map.keySet().contains(key)) {
            Tools.exit(
            "[Warning] Cetus Annotation Parser detected duplicate pragma: " +
            key);
        } else {
            cetus_map.put(key, new_str);
        }
    }

    /* When a key already exists in the map */
    @SuppressWarnings("unchecked")
    private static void addToMap(String key, Set new_set) {
        if (cetus_map.keySet().contains(key)) {
            Set set = (Set)cetus_map.get(key);
            set.addAll(new_set);
        } else {
            cetus_map.put(key, new_set);
        }
    }

    // reduction clause can be repeated as needed
    @SuppressWarnings("unchecked")
    private static void addToMap(String key, Map new_map) {
        if (cetus_map.keySet().contains(key)) {
            Map orig_map = (Map)cetus_map.get(key);
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
            cetus_map.put(key, new_map);
        }
    }

    public static enum cetus_pragma {
        cetus_parallel, 
        cetus_for, 
        cetus_threadprivate,
        cetus_use,
        cetus_def,
        cetus_private,
        cetus_firstprivate,
        cetus_lastprivate,
        cetus_reduction
    }

    public static enum cetus_clause {
        token_use,
        token_def,
        token_private,
        token_firstprivate,
        token_lastprivate,
        token_shared,
        token_reduction
    }
}

package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
* Interprocedural analysis enables computation of points-to relations through
* whole-program analysis provided by the common interprocedural framework in
* Cetus. The algorithms are based on "Context-sensitive inter-procedural
* points-to analysis in the presence of function pointers", PLDI'94. Although
* handling of function pointers and precise context-sensitivity as in the
* paper are not supported, the analysis provides single-depth
* context-sensitivity (differentiates call sites) and supports most features
* presented in the paper.
*/
public class IPPointsToAnalysis extends IPAnalysis {

    // Result of the analysis.
    private static final Map<Statement, Domain> points_to_maps =
            new HashMap<Statement, Domain>();

    // Storage for global stack.
    private Set<Symbol> global_stack;

    // Storage for abstract locations and associated sites.
    private static List<Statement> abstract_sites;
    private static List<AbstractLocation> abstract_locations;

    // Pass name
    private static final String pass_name = "[IPA:PointsTo]";

    // Calls that return a pointer to a new object.
    private static final Set<String> return_to_new = new HashSet<String>();

    // Calls that return a pointer to the first argument.
    private static final Set<String> return_to_arg1 = new HashSet<String>();

    // Calls that return a pointer to the second argument.
    private static final Set<String> return_to_arg2 = new HashSet<String>();

    // Calls that return a pointer to the third argument.
    private static final Set<String> return_to_arg3 = new HashSet<String>();

    // Calls that does not change the points-to relation (system libraries).
    private static final Set<String> safe_library_calls = new HashSet<String>();

    // Watch list of variables produced by NormalizeReturn pass.
    public static final Set<Symbol> return_vars = new HashSet<Symbol>();

    // Predefined behaviors for C99 standard library calls.
    // All of these return a "possibly" points-to relation since it is possible
    // for the call to return a null pointer if unsuccessful.
    static {
        // From C99 library
        // Calls that return a pointer to a new abstract location
        return_to_new.addAll(Arrays.asList(
                "asctime", "calloc", "ctime", "fopen", "getenv", "gmtime",
                "localtime", "malloc", "realloc", "strerror", "tmpfile",
                "tmpnam"));
        // Calls that return a pointer to the object pointed by the 1st argument
        return_to_arg1.addAll(Arrays.asList(
                "fgets", "fgetws", "gets", "memchr", "memcpy", "memmove",
                "memset", "strcat", "strchr", "strcpy", "strncat", "strncpy",
                "strpbrk", "strrchr", "strstr", "strtok", "wcscat", "wcschr",
                "wcscpy", "wcsncat", "wcsncpy", "wcspbrk", "wcsrchr", "wcsstr",
                "wcstok", "wmemchr", "wmemcpy", "wmemmove", "wmemset"));
        // Calls that return a pointer to the object pointed by the 2nd argument
        return_to_arg2.add("bsearch");
        // Calls that return a pointer to the object pointed by the 3rd argument
        return_to_arg3.add("freopen");
        // From subset of POSIX library
        return_to_new.addAll(Arrays.asList("getlogin", "ttyname"));
        return_to_arg1.add("getcwd");
        safe_library_calls.addAll(Arrays.asList(
                // sys/time.h
                "getitimer", "gettimeofday", "setitimer", "select", "utimes",
                // unistd.h
                "access", "alarm", "chdir", "chown", "close", "confstr", "dup",
                "dup2", "_exit", "execl", "execle", "execlp", "execv", "execve",
                "execvp", "faccessat", "fchdir", "fchown", "fchownat",
                "fexecve", "fork", "fpathconf", "ftruncate", "getcwd",
                "getegid", "geteuid", "getgid", "getgroups", "gethostname",
                "getlogin", "getlogin_r", "getopt", "getpgid", "getpgrp",
                "getpid", "getppid", "getsid", "getuid", "isatty", "lchown",
                "link", "linkat", "lseek", "pathconf", "pause", "pipe", "pread",
                "pwrite", "read", "readlink", "readlinkat", "rmdir", "setegid",
                "seteuid", "setgid", "setpgid", "setsid", "setuid", "sleep",
                "symlink", "symlinkat", "sysconf", "tcgetpgrp", "tcsetpgrp",
                "truncate", "ttyname", "ttyname_r", "unlink", "unlinkat",
                "write",
                // fcntl.h
                "create", "fcntl", "open", "openat"));
    }

    /** Priority for the mapping process */
    private static enum Priority {

        STRUCTDEF, STRUCTPOS, DEFINITE, POSSIBLY, DONTCARE;

        boolean isStruct() {
            return (this == STRUCTDEF || this == STRUCTPOS);
        }

        boolean isDefinite() {
            return (this == STRUCTDEF || this == DEFINITE);
        }
    }

    /**
    * AbstractLocation represents a abstract location that is not referenced to
    * in terms of a program variable.
    */
    public static class AbstractLocation extends PseudoSymbol
                                         implements Symbol {

        // List of type specifiers; standard library calls have this defined.
        private List<Specifier> heading_specifiers;

        // List of trailing specifiers - empty for all abstract locations.
        private static final List<Specifier> trailing_specifiers;

        // Associated statement.
        private Statement site;

        // Name for an abstract location.
        private String name;

        /** Fixed abstract locations */
        private static enum FIXED {

            STDERR(0), STDIN(1), STDOUT(2), ARG(3), NULL(4);

            private final int position;

            FIXED(int position) {
                this.position = position;
            }
            
            static FIXED getLocation(Expression e) {
                String name = e.toString();
                if (name.equals("stderr")) {
                     return STDERR;
                } else if (name.equals("stdin")) {
                     return STDIN;
                } else if (name.equals("stdout")) {
                     return STDOUT;
                } else if (name.equals("nullloc")) {
                     return NULL;
                } else {
                     return null;
                }
            }
            
            String getName() {
                switch (this) {
                case STDERR:
                    return "STDERR";
                case STDIN:
                    return "STDIN";
                case STDOUT:
                    return "STDOUT";
                case ARG:
                    return "ARG";
                default:
                    return "NULL";
                }
            }

            int getPosition() {
                return position;
            }
        }

        static {
            trailing_specifiers = new LinkedList<Specifier>();
        }

        /**
        * Returns an abstract locations represented by the given expression
        * {@code e}. It constructs a new one if there is no predefined location
        * associated with the given program structure {@code e}.
        * @param e the expression that is used to create a unique tag for the
        *     abstract location to be returned.
        * @return the abstract location associated with the given expression.
        */
        public static AbstractLocation getLocation(Expression e) {
            return getLocation(e, null);
        }

        /**
        * Returns an abstract locations represented by the given expression
        * {@code e} and the statement {@code s}. It performs the same operations
        * as {@link #getLocation(Expression)} does but use {@code s} as a tag
        * for the location to be returned.
        * @param e the expression from which the abstract location is created.
        * @param s the statement to be used as a tag for the location.
        * @return the abstract location associated with the given input.
        */
        public static AbstractLocation
                getLocation(Expression e, Statement s) {
            // Check for the fixed location.
            if (FIXED.getLocation(e) != null) {
                return abstract_locations.get(
                        FIXED.getLocation(e).getPosition());
            }
            if (s == null) {    // intentionally given
                s = e.getStatement();
            }
            if (s == null) {    // should not allow a location with no tag
                throw new InternalError("Abstract location doesn't have a tag");
            }
            int tag = abstract_sites.indexOf(s);
            if (tag < 0) {      // no registered site with "tag".
                abstract_locations.add(new AbstractLocation(e, s));
                tag = abstract_sites.indexOf(s);
            }
            if (tag < 0) {
                throw new InternalError("Abstract location doesn't have a tag");
            }
            return abstract_locations.get(tag);
        }

        /**
        * Dedicated constructor for system-wide fixed locations such as standard
        * stream or input argument strings.
        * @param location the type of fixed location.
        * @param main the reference IR to be used for IR search.
        */
        private AbstractLocation(FIXED location, Procedure main) {
            site = new NullStatement();
            // intentionally made it link back to the procedure so that it
            // return a valid symbol object for getAllocSymbol().
            site.setParent(main);
            abstract_sites.add(location.getPosition(), site);
            name = "<LOC#" + location.getName() + ">";
            heading_specifiers = new LinkedList<Specifier>();
            if (location.equals(FIXED.ARG)) {
                // input arguments from the program.
                heading_specifiers.add(Specifier.CHAR);
            } else if (abstract_locations.equals(FIXED.NULL)) {
                // no specifiers are necessary.
                ;
            } else {
                // standard streams.
                heading_specifiers.add(new UserSpecifier(new NameID("FILE")));
                heading_specifiers.add(PointerSpecifier.UNQUALIFIED);
            }
            abstract_locations.add(this);
        }

        /**
        * Constructs an abstract location from the given expression. Appropriate
        * actions are taken depending on the type of the given expression. For
        * now, we define two actions - for string literals and for standard
        * library calls.
        * @param e the given expression in the program.
        * @param s the reference IR to be used as a unique tag.
        */
        @SuppressWarnings("unchecked")
        private AbstractLocation(Expression e, Statement s) {
            site = s;
            abstract_sites.add(site);   // "site" shouldn't exist at this point.
            name = "<LOC#" + abstract_sites.indexOf(site) + ">";
            heading_specifiers = new LinkedList<Specifier>();
            // the function call should be a known library calls.
            if (e instanceof FunctionCall) {
                if (!StandardLibrary.contains((FunctionCall)e)) {
                    throw new InternalError("Unknown standard library calls");
                }
                List specifiers = null;
                // examine if the call is type-casted
                if (e.getParent() instanceof Typecast) {
                    specifiers = SymbolTools.getExpressionType(
                            (Expression)e.getParent());
                } else {
                    specifiers = SymbolTools.getExpressionType(e);
                }
                if (specifiers != null) {
                    heading_specifiers.addAll(specifiers);
                    // actual objects need one-level dereference
                    if (heading_specifiers.get(heading_specifiers.size()-1)
                        instanceof PointerSpecifier) {
                        heading_specifiers.remove(heading_specifiers.size()-1);
                    }
                    // removes unnecessary extern qualifier
                    heading_specifiers.remove(Specifier.EXTERN);
                }
            } else if (e instanceof StringLiteral) {
                // from string assignments
                heading_specifiers.add(Specifier.CHAR);
            } else {
                throw new InternalError("Unknown abstract location");
            }
        }

        /**
        * Returns the list of type specifiers.
        * @return the list of pre-defined type specifiers.
        */
        public List getTypeSpecifiers() {
            return heading_specifiers;
        }

        /**
        * Returns the list of array specifiers (always empty).
        * @return the list of array specifiers.
        */
        public List getArraySpecifiers() {
            return trailing_specifiers;
        }

        /**
        * Returns the name of the abstract location.
        * @return the name string.
        */
        public String getSymbolName() {
            return name;
        }

        /**
        * Returns the associated statement for the abstract location.
        * @return the statement associated with the current abstract location.
        */
        public Statement getAllocSite() {
            return site;
        }

        /**
        * Returns the IR symbol object that is most representative for the
        * abstract location.
        * @return the IR symbol
        */
        public Symbol getAllocSymbol() {
            return site.getProcedure();
        }

        /**
        * Checks if the given object is equal to the abstract location.
        * @return true if their hashcodes (names) are equal and o is an abstract
        * location.
        */
        @Override
        public boolean equals(Object o) {
            return (o instanceof AbstractLocation && name.equals(o.toString()));
        }

        /** Returns the hashcode of this abstract location. */
        @Override
        public int hashCode() {
            return name.hashCode();
        }

        /**
        * Returns the representative symbol for the abstract location. The
        * procedure object that contains the relevant <b>site</b> will be
        * returned.
        */
        @Override
        public Symbol getIRSymbol() {
            return getAllocSymbol();
        }
    }

    /**
    * Constructs a new interprocedural points-to analyzer.
    * @param program the input program.
    */
    public IPPointsToAnalysis(Program program) {
        super(program, Option.FORWARD, Option.BACKWARD, Option.TOPDOWN,
              Option.WORKLIST, Option.CONTEXT1, Option.NORMARG,
              Option.NORMRET, Option.FAILFAST);
        name = pass_name;
        abstract_sites = new LinkedList<Statement>();
        abstract_locations = new LinkedList<AbstractLocation>();
    }

    /**
    * Returns points-to relations for the given statement. The program is
    * analyzed again if the result is invalid.
    * @param stmt the statement before which points-to relations are computed.
    * @return the points-to relations map for the procedure.
    */
    public static Domain getPointsToRelations(Statement stmt) {
        // The result does not exist ==> invokes a new analysis pass.
        updatePointsToRelations(IRTools.getAncestorOfType(stmt, Program.class));
        return points_to_maps.get(stmt);
    }

    /**
    * Invalidates the analysis result. A transformation pass should trigger
    * this operation.
    */
    public static void clearPointsToRelations() {
        points_to_maps.clear();
    }

    /**
    * Recompute the static points-to relations for the given program.
    * The result of the analysis is stored as a static map for future reuse.
    * @param program the program to be analyzed.
    */
    public static void updatePointsToRelations(Program program) {
        if (points_to_maps.isEmpty()) {
            IPAnalysis analysis = new IPPointsToAnalysis(program);
            analysis.start();
        }
    }

    /**
    * Starts interprocedural points-to analysis.
    */
    @SuppressWarnings("unchecked")
    public void start() {
        // Quick return if the analysis is not possible (e.g., funcion pointer).
        if (!isAnalyzable(name)) {
            points_to_maps.putAll(PointsToAnalysis.createUniverseMap(program));
            return;
        }
        // Builds abstract stack locations.
        buildGlobalStack();
        Iterator<IPANode> iter = callgraph.topiterator();
        while (iter.hasNext()) {
            IPANode node = iter.next();
            buildLocalStack(node);
            if (verbosity >= 5) {
                PrintTools.printlnStatus(5, pass_name,
                        "LOCALS:", node.getName(), "= {",
                        PrintTools.collectionToString(getLocalStack(node),", "),
                        "}");
            }
        }
        if (verbosity >= 5) {
            PrintTools.printlnStatus(5, pass_name, "GLOBALS = {",
                    PrintTools.collectionToString(global_stack, ", "), "}");
            PrintTools.printlnStatus(5, pass_name, "Abstract Locations = {");
            for (int i = 0; i < abstract_sites.size(); i++) {
                PrintTools.printlnStatus(5, pass_name,
                        abstract_locations.get(i), abstract_sites.get(i));
            }
            PrintTools.printlnStatus(5, pass_name, "}");
        }
        // Sets the initial domain at the entry of the program.
        setRootData();
        // Invokes the solver.
        super.start();
        // Collects and stores the result.
        iter = callgraph.topiterator();
        while (iter.hasNext()) {
            IPANode node = iter.next();
            Map<Statement, Domain> result = getPTDMap(node);
            if (result != null) {
                points_to_maps.putAll(result);
                if (verbosity >= 3) {
                    //PointsToAnalysis.annotatePointsTo(
                    //        node.getProcedure(), result);
                    PrintTools.printlnStatus(3, pass_name, toPrettyDomain(
                            node.getProcedure(), result, new Integer(0)));
                }
            }
        }
    }

    /** Sets an empty domain for the root procedure node. */
    private void setRootData() {
        IPANode node = callgraph.getRoot();
        PointsToDomain in = new PointsToDomain();
        // Check if it takes arguments -- e.g., main(int argc, char *argv[])
        if (node.getParameters().size() == 2) {
            in.addRel(new PointsToRel(
                    node.getParameters().get(1),
                    abstract_locations.get(
                            AbstractLocation.FIXED.ARG.getPosition()),
                    true));
        }
        node.putData("root-data", in);
    }

    /**
    * Invokes intraprocedural analysis. The following information is the input
    * to the intra analysis.
    *  - IN data after mapping process.
    *  - OUT data after each call site.
    * @param node the procedure node to be analyzed.
    */
    @SuppressWarnings("unchecked")
    public void analyzeProcedure(IPANode node) {
        Map<Statement, Domain> result =
                PointsToAnalysis.getPointsToRelations(node);
        putPTDMap(result, node);
        if (verbosity >= 5) {
            PrintTools.printlnStatus(5, pass_name, 
                toPrettyDomain(node.getProcedure(), result, new Integer(0)));
        }
    }

    /** Adds extra points-to relations from argument normalization. */
    private static Domain
            addTempArguments(CallSite call_site, Domain caller_in) {
        PrintTools.printlnStatus(3, pass_name, "CALL-IN =", caller_in);
        Domain ret = caller_in;
        List<Traversable> temp_assigns =
                call_site.getTempAssignments().getChildren();
        for (int i = 0; i < temp_assigns.size(); i++) {
            Traversable temp_assign = temp_assigns.get(i);
            if (temp_assign instanceof ExpressionStatement) {
                // Should be guaranteed
                AssignmentExpression assign =
                        (AssignmentExpression)temp_assign.getChildren().get(0);
                ret = PointsToAnalysis.processBasicAssignment(
                        assign.getLHS(), assign.getRHS(),
                        call_site.getFunctionCall().getStatement(), ret);
            }
            if (ret instanceof PointsToDomain.Universe) {
                break;
            }
        }
        PrintTools.printlnStatus(3, pass_name, "(+) TEMP =", ret);
        return ret;
    }

    /**
    * Removes extra points-to relations from argument normalization.
    * @param call_site the relevant call site.
    * @param caller_in the points-to information to be processed. 
    */
    private static PointsToDomain
            removeTempArguments(CallSite call_site, PointsToDomain caller_in) {
        PointsToDomain ret = caller_in.clone();
        ret.keySet().removeAll(
                SymbolTools.getSymbols(call_site.getTempAssignments()));
        return ret;
    }

    /**
    * Performs mapping process which converts caller's domain to callee's
    * domain.
    * @param call_site the calling site.
    * @param caller_in the caller's domain available before the call site.
    * @return the resulting callee's in data at the entry to the callee.
    */
    private static PointsToDomain
            mapProcess(CallSite call_site, PointsToDomain caller_in) {
        Map<Symbol, Set<Symbol>> map_info =
                new LinkedHashMap<Symbol, Set<Symbol>>();
        PointsToDomain ret = new PointsToDomain();
        IPANode callee_node = call_site.getCallee();
        if (callee_node == null || callee_node.containsVarArg()) {
            // TODO: this should not be reachable now. handling of vararg could
            // be added in the future.
            throw new InternalError("variable arguments are not handled");
        }
        List<Symbol> params = call_site.getParameters();
        List<Expression> args = call_site.getNormArguments();
        if (params.size() != args.size()) {
            throw new InternalError(
                    "mismatching size in parameters and arguments");
        }
        PrintTools.printlnStatus(3, pass_name, "MAP-PROCESS:", call_site);
        // The priority of the map process:
        // STRUCTDEF: struct/definite
        // STRUCTPOS: struct/possibly
        // DEFINITE : non-struct/definite
        // POSSIBLY : non-struct/possibly
        mapSymbols(params, args, ret, caller_in, map_info, callee_node,
                Priority.STRUCTDEF);
        mapSymbols(params, args, ret, caller_in, map_info, callee_node,
                Priority.STRUCTPOS);
        mapSymbols(params, args, ret, caller_in, map_info, callee_node,
                Priority.DEFINITE);
        mapSymbols(params, args, ret, caller_in, map_info, callee_node,
                Priority.POSSIBLY);
        if (verbosity >= 3) {
            PrintTools.printlnStatus(3, pass_name,
                    "MAP-INFO:", mapInfoToString(map_info));
        }
        // Update map_info and work queue of callers for intra analysis.
        Map<CallSite, Map<Symbol, Set<Symbol>>> map_infos =
                getMapInfos(callee_node);
        if (map_infos == null) {
            map_infos = new LinkedHashMap<CallSite, Map<Symbol, Set<Symbol>>>();
            putMapInfos(map_infos, callee_node);
        }
        // Update the outstanding map-info.
        map_infos.put(call_site, map_info);
        PrintTools.printlnStatus(3, pass_name,
                "MAP-RESULT:", caller_in, "=>", ret);
        return ret;
    }

    /**
    * Performs mapping of symbols with the given parameters/arguments and global
    * information. This method is called with two different priorities for
    * struct variables.
    * @param params the list of parameter symbols.
    * @param args the list of matching argument expressions.
    * @param ret the points-to domain to be returned.
    * @param in the given input points-to domain.
    * @param map_info the extra mapping information for invisible variables.
    * @param node the procedure node being called.
    * @param process_struct the priority for struct variables.
    */
    private static void mapSymbols(List<Symbol> params,
                                   List<Expression> args,
                                   PointsToDomain ret,
                                   PointsToDomain in,
                                   Map<Symbol, Set<Symbol>> map_info,
                                   IPANode node, Priority priority) {
        // Process parameters.
        for (int i = 0; i < params.size(); i++) {
            // args have been normalized so that it contain either ID or Literal
            if (!(args.get(i) instanceof Identifier)) {
                continue;
            }
            Symbol arg_i = ((Identifier)args.get(i)).getSymbol();
            Symbol param_i = params.get(i);
            mapSymbol(param_i, arg_i, ret, in, map_info, node, priority);
        }
        // Process globals.
        for (Symbol global : in.keySet()) {
            if (getGlobalStack(node).contains(global) &&
                !(global instanceof DerefSymbol)) {
                 mapSymbol(global, global, ret, in, map_info, node, priority);
            }
        }
    }

    /** For debug */
    private static String mapInfoToString(Map<Symbol, Set<Symbol>> map) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        Iterator<Symbol> iter = map.keySet().iterator();
        for (int i = 0; iter.hasNext(); i++) {
            Symbol key = iter.next();
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(key.getSymbolName()).append("=>{");
            sb.append(PrintTools.collectionToString(map.get(key), ","));
            sb.append("}");
        }
        sb.append("]");
        return sb.toString();
    }

    /**
    * Performs mapping process recursively. This method is based on the thesis
    * written by Mariam Emami.
    * @param callee_var the callee's variable.
    * @param caller_var the caller's variable.
    * @param callee_in the resulting mapped domain.
    * @param caller_in the caller's domain before the call site.
    * @param map_info the mapping information from invisibles to caller's vars.
    * @param node the current callee procedure.
    * @param process_struct the priority for struct variables.
    */
    private static void mapSymbol(Symbol callee_var,
                                  Symbol caller_var,
                                  PointsToDomain callee_in,
                                  PointsToDomain caller_in,
                                  Map<Symbol, Set<Symbol>> map_info,
                                  IPANode node, Priority priority) {
        // No relation is found or the symbol is not a pointer.
        if (caller_in.get(caller_var) == null ||
                (!SymbolTools.isPointer(callee_var) &&
                !SymbolTools.isPointerParameter(callee_var))) {
            return;
        }
        Set<Symbol> local_stack = getLocalStack(node);
        Set<Symbol> global_stack = getGlobalStack(node);
        for (PointsToRel rel : caller_in.get(caller_var)) {
            Symbol pointed_to = rel.getPointedToSymbol();
            // Priority checking.
            if (!(priority.isDefinite() == rel.isDefinite() &&
                    priority.isStruct() ==
                    SymbolTools.isStruct(pointed_to, node.getProcedure()) ||
                    priority == Priority.DONTCARE)) {
                continue;
            }
            // pointed_to is in the callee's scope.
            if (local_stack.contains(pointed_to) ||
                    global_stack.contains(pointed_to)) {
                callee_in.addRel(new PointsToRel(
                        callee_var, pointed_to, rel.isDefinite()));
                // follow-up process should ignore the priority.
                mapSymbol(pointed_to, pointed_to, callee_in, caller_in,
                        map_info, node, Priority.DONTCARE);
            // pointed_to is not in the callee's scope.
            } else {
                Symbol inv_var = getExistingInvisibleVar(pointed_to, map_info);
                if (inv_var == null) {
                    inv_var = defineInvisibleVar(callee_var, node);
                    assert inv_var != null:
                            "failed to define invisible variable.";
                    addMapInfo(inv_var, pointed_to, map_info);
                    if (priority.isStruct()) {
                        addStructMapInfo(inv_var, pointed_to, map_info,
                                local_stack);
                    }
                }
                callee_in.addRel(new PointsToRel(
                        callee_var, inv_var, rel.isDefinite()));
                // follow-up process should ignore the priority.
                mapSymbol(inv_var, pointed_to, callee_in, caller_in, map_info,
                        node, Priority.DONTCARE);
            }
        }
    }

    /**
    * Return the invisible variable in "map_info" for "var".
    * @param var the given variable (caller's variable).
    * @param map_info the given map-info (callee var to caller vars).
    * @return the callee's invisible variable mapped to "var".
    */
    private static Symbol getExistingInvisibleVar(
            Symbol var, Map<Symbol, Set<Symbol>> map_info) {
        for (Symbol inv_var : map_info.keySet()) {
            Set<Symbol> caller_vars = map_info.get(inv_var);
            if (caller_vars.contains(var)) {
                return inv_var;
            }
        }
        return null;
    }

    /**
    * Returns the an invisible variable for "var".
    * @param var the reference variable.
    * @param node the relevant procedure node.
    */
    private static Symbol defineInvisibleVar(Symbol var, IPANode node) {
        Symbol ret = DerefSymbol.get(var);
        Set<Symbol> global_stack = getGlobalStack(node);
        Set<Symbol> local_stack = getLocalStack(node);
        // Strictly checks if the invisible symbol is a valid location.
        if (!global_stack.contains(ret) && !local_stack.contains(ret)) {
            ret = null;
        }
        return ret;
    }

    /** Accessors and modifiers of the data kept in the specified node. */
    private static Set<Symbol> getLocalStack(IPANode node) {
        Set<Symbol> ret = node.getData("local-stack-set");
        return ret;
    }

    private static Set<Symbol> getGlobalStack(IPANode node) {
        Set<Symbol> ret = node.getData("global-stack-set");
        return ret;
    }

    private static Map<CallSite, Map<Symbol, Set<Symbol>>>
            getMapInfos(IPANode node) {
        Map<CallSite, Map<Symbol, Set<Symbol>>> ret = node.getData("map-infos");
        return ret;
    }

    private static Map<Statement, Domain> getPTDMap(IPANode node) {
        Map<Statement, Domain> ret = node.getData("ptd-map");
        return ret;
    }

    private static void putLocalStack(Set<Symbol> stack, IPANode node) {
        node.putData("local-stack-set", stack);
    }

    private static void putGlobalStack(Set<Symbol> stack, IPANode node) {
        node.putData("global-stack-set", stack);
    }

    private static void putMapInfos(
            Map<CallSite, Map<Symbol, Set<Symbol>>> infos, IPANode node) {
        node.putData("map-infos", infos);
    }

    private static void
            putPTDMap(Map<Statement, Domain> ptdmap, IPANode node) {
        node.putData("ptd-map", ptdmap);
    }

    private static void
            putReturnRelations(Set<PointsToRel> rels, CallSite callsite) {
        IPANode callee = callsite.getCallee();
        Map<CallSite, Set<PointsToRel>> return_rel = getReturnRelations(callee);
        if (return_rel == null) {
            return_rel = new LinkedHashMap<CallSite, Set<PointsToRel>>();
            callee.putData("return-rel", return_rel);
        }
        Set<PointsToRel> context_rel = return_rel.get(callsite);
        if (context_rel == null) {
            context_rel = new LinkedHashSet<PointsToRel>();
            return_rel.put(callsite, context_rel);
        }
        context_rel.addAll(rels);
    }

    private static Map<CallSite, Set<PointsToRel>>
            getReturnRelations(IPANode node) {
        Map<CallSite, Set<PointsToRel>> ret = node.getData("return-rel");
        return ret;
    }

    /**
    * Returns the set of points-to relations pointed by the return value of
    * the specified function call. It uses the normalized return value as the
    * pointer symbol and the caller's symbol as the pointed-to symbol. For a
    * standard library call, the function symbol is used as the pointer symbol.
    * 
    * @param callsite the calling site that generates the query.
    * @return the set of return points-to relations. an empty set means there
    * is no relations created, and {@code null} means no information is
    * available (should produce universe doamin).
    */
    public static Set<PointsToRel> getReturnRelations(CallSite callsite) {
        Set<PointsToRel> ret = null;
        IPANode callee = callsite.getCallee();
        if (callee == null) {
            PointsToRel lib_rel = getStandardLibraryReturnRelation(callsite);
            if (lib_rel != null) {
                ret = new HashSet<PointsToRel>();
                // Skip dummy relation meaning no relation is newly created.
                if (!lib_rel.getPointerSymbol().equals(
                        lib_rel.getPointedToSymbol())) {
                    ret.add(lib_rel);
                }
            }
        } else {
            if (callee.getVisitCount() == 0) {
                // the callee was not visited at all.
                ret = new HashSet<PointsToRel>();
            } else {
                Map<CallSite, Set<PointsToRel>> relations =
                        getReturnRelations(callee);
                if (relations != null) {
                    ret = relations.get(callsite);
                }
            }
        }
        return ret;
    }

    /**
    * Checks if the given function call is known for having no side effects on
    * the "points-to relation" (not on memory).
    * @param fcall the function call to be examined.
    * @return true if it is either a standard library call or a known system
    * library call.
    */
    public static boolean isSafeLibraryCall(FunctionCall fcall) {
        return (StandardLibrary.contains(fcall) ||
                safe_library_calls.contains(fcall.getName().toString()));
    }

    /**
    * Adds key:value mapping in the map_info.
    * @param key the variable used as key (invisible variable).
    * @param value the variable used as value (caller's variable).
    * @param map_info the given map-info.
    */
    private static boolean addMapInfo(
            Symbol key, Symbol value, Map<Symbol, Set<Symbol>> map_info) {
        Set<Symbol> value_set = map_info.get(key);
        if (value_set == null) {
            value_set = new LinkedHashSet<Symbol>();
            map_info.put(key, value_set);
        }
        value_set.add(value);
        return true;
    }

    /**
    * Adds struct map info by replacing the key with value in the abstract
    * stack.
    */
    private static void addStructMapInfo(Symbol key,
                                         Symbol value,
                                         Map<Symbol,
                                         Set<Symbol>> map_info,
                                         Set<Symbol> stack) {
        for (Symbol sym : stack) {
            if (sym instanceof AccessSymbol) {
                AccessSymbol struct_sym = (AccessSymbol)sym;
                if (struct_sym.getBaseSymbol().equals(key)) {
                    addMapInfo(struct_sym, new AccessSymbol(
                            value, struct_sym.getMemberSymbol()), map_info);
                }
            }
        }
    }

    /** Returns the map-info associated with the given context. */
    private static Map<Symbol, Set<Symbol>>
            getMapInfo(CallSite context, IPANode node) {
        Map<CallSite, Map<Symbol, Set<Symbol>>> map_infos = getMapInfos(node);
        if (map_infos == null) {
            return null;
        } else {
            return map_infos.get(context);
        }
    }

    /**
    * Updates the forward information for each call site by performing mapping,
    * and by maintaining map-info for each different single-depth context.
    * @param node the procedure node being analyzed.
    * @return the set of call sites whose data has been changed.
    */
    public Set<CallSite> updateCalls(IPANode node) {
        Set<CallSite> ret = new LinkedHashSet<CallSite>();
        Map<Statement, Domain> ptd_map = getPTDMap(node);
        for (CallSite call_site : node.getCallSites()) {
            Domain curr_data = ptd_map.get(call_site.getStatement());
            PrintTools.printlnStatus(3, name, call_site);
            PrintTools.printlnStatus(3, name, "prev-call-in =", call_site.in());
            PrintTools.printlnStatus(3, name, "curr-call-in =", curr_data);
            if (!call_site.in().equals(curr_data)) {
                ret.add(call_site);
                call_site.in(curr_data);
                IPANode callee = call_site.getCallee();
                // library calls.
                if (callee == null) {
                    // user library calls or universe IN domain.
                    if (call_site.in()instanceof PointsToDomain.Universe ||
                            (!StandardLibrary.contains(
                            call_site.getFunctionCall()) &&
                            !safe_library_calls.contains(
                            call_site.getName()))) {
                        call_site.out(PointsToDomain.Universe.getUniverse());
                    // standard library calls with non-universe IN domain.
                    } else {
                        call_site.out(call_site.in().clone());
                        // extra safey check for functions returning a pointer.
                        if (StandardLibrary.contains(
                                call_site.getFunctionCall()) &&
                                getReturnRelations(call_site) == null) {
                            call_site.out(
                                    PointsToDomain.Universe.getUniverse());
                        }
                    }
                // difficult cases to handle.
                } else if (callee.containsVarArg()) {
                    callee.in(call_site, PointsToDomain.Universe.getUniverse());
                // normal cases.
                } else {
                    // it is better to clarify what is passed to callee for
                    // these.
                    if (call_site.in() instanceof NullDomain ||
                        call_site.in() instanceof PointsToDomain.Universe) {
                        callee.in(call_site, call_site.in());
                    } else {
                        Domain callee_in =
                                addTempArguments(call_site, curr_data);
                        if (callee_in instanceof PointsToDomain) {
                            callee_in = mapProcess(
                                    call_site, (PointsToDomain) callee_in);
                            // we need to merge the current data with any
                            // existing in data to the callee to reflect the
                            // effect of different call paths above the direct
                            // caller.
                            if (callee.in(call_site) instanceof PointsToDomain)
                                callee_in =
                                        callee.in(call_site).merge(callee_in);
                        } else { // only universe is possible here.
                            call_site.in(callee_in);
                        }
                        callee.in(call_site, callee_in);
                    }
                }
            }
        }
        return ret;
    }

    /**
    * Not used.
    */
    public boolean updateCall(IPANode node) {
        // Not used.
        return false;
    }

    /**
    * Updates the backward information available at the exit of a procedure by
    * performing unmapping.
    * @param node the procedure node to be analyzed.
    * @return true if there is any state change, false otherwise.
    */
    public boolean updateReturn(IPANode node) {
        if (node.getCallingSites().isEmpty()) {
            return false;
        }
        // Computes callee_out.
        Domain callee_out = null;
        Map<Statement, Domain> ptd_map = getPTDMap(node);
        List<ReturnStatement> return_stmts = IRTools.getStatementsOfType(
                node.getProcedure(), ReturnStatement.class);
        assert ! return_stmts.isEmpty() : "no return statements found.";
        for (Statement stmt : return_stmts) {
            if (callee_out == null) {
                callee_out = ptd_map.get(stmt);
            } else {
                callee_out =
                        PointsToDomain.merge(callee_out, ptd_map.get(stmt));
            }
        }
        PrintTools.printlnStatus(3, name, "callee_out =", callee_out);
        CallSite context = getContext(node);
        // Exceptional case handling
        if (callee_out instanceof PointsToDomain.Universe) {
            if (context.out() instanceof PointsToDomain.Universe) {
                return false;
            }
            context.out(callee_out);
            return true;
        } else if (callee_out instanceof NullDomain) {
            if (context.out() instanceof NullDomain) {
                return false;
            }
            // this should not be allowed.
            throw new InternalError(
                    "Infeasible states found during updateReturn");
        }
        // callee_out is a points-to domain.
        if (context.in() instanceof PointsToDomain.Universe) {
            if (context.out() instanceof PointsToDomain.Universe) {
                return false;
            }
            context.out(context.in());
            return true;
        } else if (context.in()instanceof NullDomain) {
            throw new InternalError("Infeasible intra analysis was performed");
        }
        // context.in() now should be points-to domain.
        // caller_out will be updated reflecting KILL and GEN.
        PointsToDomain caller_in = null;
        PointsToDomain caller_out = new PointsToDomain();
        // Computes caller_out U= (callee_out - kill).
        caller_in = (PointsToDomain)context.in();
        for (Symbol key : caller_in.keySet()) {
            for (PointsToRel pt_rel : caller_in.get(key)) {
                PointsToRel callee_rel = getCalleePTR(pt_rel, node);
                // a pt_rel in caller_in is intact only if its callee version
                // remains unchanged in callee_out or its callee version does
                // not exist due to no passed information (local only to the
                // caller).
                if (callee_rel == null ||
                    callee_out instanceof PointsToDomain &&
                    ((PointsToDomain)callee_out).containsPTR(callee_rel)) {
                    caller_out.addRel(pt_rel.clone());
                }
            }
        }
        PrintTools.printlnStatus(3, name, "(-) KILL =", caller_out);
        // Computes caller_out U= GEN.
        if (callee_out != null && callee_out instanceof PointsToDomain) {
            PointsToDomain callee_in = (PointsToDomain)node.in(context);
            for (Symbol key : ((PointsToDomain)callee_out).keySet()) {
                for (PointsToRel pt_rel:((PointsToDomain)callee_out).get(key)) {
                    // 1. Process "D"efinitely points-to relations.
                    if (pt_rel.isDefinite()) {
                        if (!callee_in.containsPTR(pt_rel)) {
                            PointsToRel caller_ptr = getCallerPTR(pt_rel, node);
                            if (caller_ptr != null) {
                                caller_out.addRel(caller_ptr);
                            }
                        }
                    // 2. Process "P"ossibly points-to relations.
                    } else {
                        PointsToRel caller_ptr = getCallerPTR(pt_rel, node);
                        if (caller_ptr != null) {
                            caller_out.addRel(caller_ptr);
                        }
                    }
                }
            }
        }
        PrintTools.printlnStatus(3, name, "(+) GEN =", caller_out);
        caller_out = removeTempArguments(context, caller_out);
        PrintTools.printlnStatus(3, name, "(-) TEMP =", caller_out);
        PrintTools.printlnStatus(3, name, "prev-call-out =", context.out());
        // Collects relations pointed by the return value.
        boolean return_changed = false;
        if (callee_out instanceof PointsToDomain &&
            SymbolTools.isPointer(node.getProcedure())) {
            PointsToDomain return_ptd = (PointsToDomain)callee_out;
            Set<PointsToRel> return_rels = new LinkedHashSet<PointsToRel>();
            for (Symbol key : ((PointsToDomain)callee_out).keySet()) {
                if (return_vars.contains(key)) {
                    Set<PointsToRel> rels = return_ptd.get(key);
                    for (PointsToRel rel : rels) {
                        Symbol pointed_to = fromCalleeToCaller(
                                rel.getPointedToSymbol(), node);
                        if (pointed_to != null) {
                            return_rels.add(new PointsToRel(
                                    key, pointed_to, rel.isDefinite()));
                        }
                    }
                }
            }
            if (!return_rels.equals(getReturnRelations(context))) {
                return_changed = true;
                putReturnRelations(return_rels, context);
                PrintTools.printlnStatus(3, name,
                        "return relations:", context, "=>", return_rels);
            }
        }
        // Compensate context insensitivity beyond direct caller.
        /* TODO: check if this speeds up the analysis
        if (context.out() instanceof PointsToDomain &&
            caller_out instanceof PointsToDomain) {
            caller_out = (PointsToDomain)caller_out.merge(context.out());
        }
        */
        if (!context.out().equals(caller_out) || return_changed) {
            if (callee_out instanceof PointsToDomain.Universe) {
                context.out(callee_out);
            } else {
                context.out(caller_out);
            }
            return true;
        }
        return false;
    }

    /** Returns a caller variable corresponding to the given callee variable. */
    private static Symbol
            fromCalleeToCaller(Symbol callee_var, IPANode node) {
        // Check directly accessed global variables.
        // Other cases are handled following map info. 
        if (getGlobalStack(node).contains(callee_var) &&
            (callee_var instanceof Traversable ||
             callee_var instanceof AbstractLocation)) {
            return callee_var;
        }
        CallSite context = getContext(node);
        // Check map-info.
        // One callee variable may correspond to multiple caller variables, so
        // the first element (visited first by iterator) is returned.
        Map<Symbol, Set<Symbol>> map_info = getMapInfo(context, node);
        Set<Symbol> caller_vars = map_info.get(callee_var);
        if (caller_vars != null && !caller_vars.isEmpty()) {
            return caller_vars.iterator().next();
        }
        // Check parameter - recursively handles struct or invisible symbols.
        if (callee_var instanceof AccessSymbol) {
            AccessSymbol str_symbol = (AccessSymbol) callee_var;
            Symbol base = fromCalleeToCaller(str_symbol.getBaseSymbol(), node);
            if (base != null) {
                return new AccessSymbol(base, str_symbol.getMemberSymbol());
            }
        } else if (callee_var instanceof DerefSymbol) {
            DerefSymbol inv_symbol = (DerefSymbol) callee_var;
            Symbol ref = fromCalleeToCaller(inv_symbol.getRefSymbol(), node);
            if (ref != null) {
                return DerefSymbol.get(ref);
            }
        }
        int position = context.getParameters().indexOf(callee_var);
        if (position >= 0) {
            Expression arg = context.getNormArguments().get(position);
            assert arg instanceof Identifier : "argument not normalized";
            return PointsToAnalysis.exprToLocation(arg);
        }
        return null;
    }

    /**
    * Returns the callee variable corresponding to the given caller variable.
    */
    private static Symbol
            fromCallerToCallee(Symbol caller_var, IPANode node) {
        // Check global.
        if (getGlobalStack(node).contains(caller_var)) {
            return caller_var;
        }
        // Check argument.
        CallSite context = getContext(node);
        for (int i = 0; i < context.getNormArguments().size(); i++) {
            Expression arg = context.getNormArguments().get(i);
            Symbol symbol = PointsToAnalysis.exprToLocation(arg);
            if (caller_var.equals(symbol)) {
                return context.getParameters().get(i);
            }
        }
        // Check map-info.
        Map<Symbol, Set<Symbol>> map_info = getMapInfo(context, node);
        if (map_info != null) {
            for (Symbol callee_var : map_info.keySet()) {
                if (map_info.get(callee_var).contains(caller_var)) {
                    return callee_var;
                }
            }
        }
        return null;
    }

    /** Returns the caller's view of the given callee_ptr. */
    private static PointsToRel
            getCallerPTR(PointsToRel callee_ptr, IPANode node) {
        Symbol pointer =
                fromCalleeToCaller(callee_ptr.getPointerSymbol(), node);
        Symbol pointee =
                fromCalleeToCaller(callee_ptr.getPointedToSymbol(), node);
        if (pointer == null || pointee == null) {
            return null;
        }
        return new PointsToRel(pointer, pointee, callee_ptr.isDefinite());
    }

    /** Returns the callee's view of the given caller_ptr. */
    private static PointsToRel
            getCalleePTR(PointsToRel caller_ptr, IPANode node) {
        PointsToRel ret = null;
        Symbol pointer =
                fromCallerToCallee(caller_ptr.getPointerSymbol(), node);
        Symbol pointee =
                fromCallerToCallee(caller_ptr.getPointedToSymbol(), node);
        if (pointer != null && pointee != null) {
            ret = new PointsToRel(pointer, pointee, caller_ptr.isDefinite());
        }
        // Let other cases return null. This situation can happen if either the
        // pointer or the pointed symbol is not accessible at all in the callee.
        return ret;
    }

    /**
    * Adds the given symbol to the specified stack location while keeping track
    * of any aggregate types. The last parameter "types" is used to detect any
    * recursively defined aggregate types.
    * @param symbol the symbol to be added. 
    * @param stack the set of stack locations to be updated.
    * @param tr the traversable object to be searched for any aggregate type.
    * @param types the list of previously added types.
    */
    private static void addStackLocation(
            Symbol symbol, Set<Symbol> stack, Traversable tr, List<List> types){
        List<Specifier> type = getTypeSpecifiers(symbol);
        // Empty types need not to be handled.
        if (type.isEmpty()) {
            return;
        }
        ClassDeclaration cdecl = SymbolTools.getClassDeclaration(symbol, tr);
        // Avoid recursive aggregate type.
        if (cdecl != null && types.contains(type)) {
            return;
        }
        stack.add(symbol);
        types.add(type);
        Symbol deref_symbol = symbol;
        while ((deref_symbol = DerefSymbol.get(deref_symbol)) != null) {
            stack.add(deref_symbol);
            type = getTypeSpecifiers(deref_symbol);
            types.add(type);
            if (cdecl != null) {
                for (Symbol member : SymbolTools.getSymbols(cdecl)) {
                    addStackLocation(new AccessSymbol(deref_symbol, member),
                                     stack, tr, types);
                }
            }
        }
    }

    /**
    * Returns the type specifiers after handling array parameters and discarding
    * useless information such as EXTERN.
    */
    protected static List<Specifier> getTypeSpecifiers(Symbol symbol) {
        List<Specifier> ret = new LinkedList<Specifier>();
        for (Object type : symbol.getTypeSpecifiers()) {
            if (!type.equals(Specifier.EXTERN)) {
                ret.add((Specifier)type);
            }
        }
        // Additional pointer specifier is added only for "formal" parameters.
        if (symbol instanceof Traversable &&
            SymbolTools.isFormal(symbol) &&
            !symbol.getArraySpecifiers().isEmpty()) {
            ret.add(PointerSpecifier.UNQUALIFIED);
        }
        return ret;
    }

    /** Builds the set of global abstract stack locations for the program. */
    private void buildGlobalStack() {
        global_stack = new LinkedHashSet<Symbol>();
        for (Object o : program.getChildren()) {
            TranslationUnit tu = (TranslationUnit)o;
            for (Symbol tu_symbol : SymbolTools.getVariableSymbols(tu)) {
                addStackLocation(
                        tu_symbol, global_stack, tu, new LinkedList<List>());
            }
        }
        // Adds fixed abstract locations.
        for (AbstractLocation.FIXED location :
                EnumSet.allOf(AbstractLocation.FIXED.class)) {
            addStackLocation(new AbstractLocation(
                    location, callgraph.getRoot().getProcedure()), global_stack,
                    callgraph.getRoot().getProcedure(), new LinkedList<List>());
        }
    }

    /** Builds the set of local abstract stack locations for the given node. */
    private void buildLocalStack(IPANode node) {
        Procedure proc = node.getProcedure();
        Set<Symbol> local_stack = new LinkedHashSet<Symbol>();
        // Formal parameters.
        for (Object o : proc.getParameters()) {
            VariableDeclaration vdecl = (VariableDeclaration)o;
            if (vdecl.getNumDeclarators() == 1) {
                if (vdecl.toString().equals("void ")) {  // contains("void")?
                    continue;
                }
                Symbol param = (Symbol)vdecl.getChildren().get(0);
                addStackLocation(
                        param, local_stack, proc, new LinkedList<List>());
            } else {
                PrintTools.printlnStatus(0, name,
                        "[WARNING] multiple declarators in a parameter");
            }
        }
        // Automatic variables and dynamically allocated locations.
        DFIterator<Traversable> iter = new DFIterator<Traversable>(proc);
        while (iter.hasNext()) {
            Traversable t = iter.next();
            // Symbols declared in a compound statement.
            if (t instanceof CompoundStatement) {
                SymbolTable st = (SymbolTable)t;
                for (Symbol symbol : SymbolTools.getVariableSymbols(st)) {
                    addStackLocation(
                            symbol, local_stack, proc, new LinkedList<List>());
                }
            // Abstract locations assigned by standard library calls.
            } else if (t instanceof FunctionCall &&
                    return_to_new.contains(
                    ((FunctionCall)t).getName().toString())) {
                AbstractLocation heap =
                        AbstractLocation.getLocation((Expression)t);
                addStackLocation(
                        heap, global_stack, proc, new LinkedList<List>());
            // Abstract locations assigned by string literals.
            } else if (t instanceof StringLiteral &&
                    (t.getParent() instanceof Initializer ||
                    t.getParent() instanceof AssignmentExpression)) {
                AbstractLocation string =
                        AbstractLocation.getLocation((Expression)t);
                addStackLocation(
                        string, global_stack, proc, new LinkedList<List>());
            }
        }
        // Temporary identifiers for the normalized arguments.
        for (CallSite call_site : node.getCallSites()) {
            for (Symbol temp_args :
                    SymbolTools.getSymbols(call_site.getTempAssignments())) {
                addStackLocation(
                        temp_args, local_stack, proc, new LinkedList<List>());
            }
        }
        // Stores the completed stack locations in the node.
        putLocalStack(local_stack, node);
        putGlobalStack(global_stack, node);
    }

    /**
    * Returns a points-to relation from the given call site that contains a
    * standard library call.
    * @param callsite the call site containing a standard library call.
    * @return a points-to relation if possible, null otherwise. A dummy relation
    * is returned for a case that does not create any relations.
    */
    private static PointsToRel
            getStandardLibraryReturnRelation(CallSite callsite) {
        PointsToRel ret = null;
        if (!StandardLibrary.contains(callsite.getFunctionCall())) {
            return ret;
        }
        Expression fname = callsite.getFunctionCall().getName();
        Symbol pointer = SymbolTools.getSymbolOf(fname);
        Symbol pointed_to = null;
        if (return_to_arg1.contains(fname.toString())) {
            Symbol arg =
                    SymbolTools.getSymbolOf(callsite.getArguments().get(0));
            if (arg != null) {
                pointed_to = DerefSymbol.get(arg);
            }
        } else if (return_to_arg2.contains(fname.toString())) {
            Symbol arg =
                    SymbolTools.getSymbolOf(callsite.getArguments().get(1));
            if (arg != null) {
                pointed_to = DerefSymbol.get(arg);
            }
        } else if (return_to_arg3.contains(fname.toString())) {
            Symbol arg =
                    SymbolTools.getSymbolOf(callsite.getArguments().get(2));
            if (arg != null) {
                pointed_to = DerefSymbol.get(arg);
            }
        } else if (return_to_new.contains(fname.toString())) {
            pointed_to =
                    AbstractLocation.getLocation(callsite.getFunctionCall());
        } else {
            pointed_to = pointer;       // add a dummy for other cases.
        }
        if (pointed_to != null) {
            ret = new PointsToRel(pointer, pointed_to, false);
        }
        return ret;
    }

    /**
    * Checks if the current points-to analysis result contains
    * points-to-universe relation.
    */
    public static boolean containsUniverse() {
        return points_to_maps.containsValue(
                PointsToDomain.Universe.getUniverse());
    }
}

package cetus.hir;

import java.util.*;

/**
* <b>SymbolTools</b> provides tools for interfacing symbol tables and related
* searches.
*/
public final class SymbolTools {

    private SymbolTools() {
    }

    /**
    * Makes links from all {@link IDExpression} objects in the program to
    * their corresponding declarators while generating warnings if there is
    * any undeclared variables or functions. This method is called with the
    * whole program, before any Cetus passes by default and provides a short cut
    * to the declaration point, which enables faster access to the declaration
    * when necessary. Pass writers can call this routine after changing a
    * certain part of the program, e.g., a specific program section within a
    * scope, to reflect the change due to new insertion of declaration.
    * @param t the input cetus IR.
    */
    public static void linkSymbol(Traversable t) {
        int[] num_updates = new int[]{ 0 };
        double timer = Tools.getTime();
        DFIterator<Enumeration> enum_iter =
                new DFIterator<Enumeration>(t, Enumeration.class);
        enum_iter.pruneOn(ExpressionStatement.class);
        enum_iter.pruneOn(VariableDeclaration.class);
        while (enum_iter.hasNext()) {
            Enumeration en = enum_iter.next();
            SymbolTable st = IRTools.getAncestorOfType(en, SymbolTable.class);
            addSymbols(st, en);
        }
        DFIterator<Identifier> id_iter =
                new DFIterator<Identifier>(t, Identifier.class);
        while (id_iter.hasNext()) {
            searchAndLink(id_iter.next(), num_updates);
        }
        String msg = String.format("%d updates in %.2f seconds",
                                   num_updates[0], Tools.getTime(timer));
        PrintTools.printlnStatus(0, "[LinkSymbol]", msg);
    }

    /**
    * Searches for the declaration of the specified identifier and makes a link
    * from the identifier to the declaration.
    */
    private static void searchAndLink(Identifier id, int[] num_updates) {
        String id_name = id.getName();
        // These cases are skipped intentionally.
        if (id.getParent() instanceof Declarator ||    // it is a Symbol object.
            id.getParent() instanceof GotoStatement || // not a variable.
            id.getParent() instanceof Label ||         // not a variable.
            id_name.equals("") ||                      // void Symbol.
            id_name.equals("__PRETTY_FUNCTION__") ||   // gcc keyword.
            id_name.equals("__FUNCTION__") ||          // gcc keyword.
            id_name.startsWith("__builtin")) {         // gcc keyword.
            return;
        }
        Declaration decl = searchDeclaration(id);
        if (decl == null) {
            return;
        }
        // Lookup the symbol object in the declaration and add links.
        if (decl instanceof Procedure) {
            id.setSymbol((Symbol)decl);
            num_updates[0]++;
        } else if (decl instanceof VariableDeclaration ||
                   decl instanceof Enumeration) {
            List<Traversable> children = decl.getChildren();
            int children_size = children.size();
            Declarator d = null;
            for (int i = 0; i < children_size; i++) {
                Traversable child = children.get(i);
                if (id.equals(((Declarator)child).getID())) {
                    d = (Declarator)child;
                    id.setSymbol((Symbol)d);
                    num_updates[0]++;
                    break;
                }
            }
        }
    }

    // Returns the type of an expression
    @SuppressWarnings("unchecked")
    private static List getType(Traversable e) {
        if (!(e instanceof Expression)) {
            return null;
        }
        if (e instanceof Identifier) {
            Symbol var = ((Identifier)e).getSymbol();
            if (var == null) {
                return null;
            } else {
                return var.getTypeSpecifiers();
            }
        } else if (e instanceof AccessExpression) {
            return getType(((AccessExpression)e).getRHS());
        } else if (e instanceof ArrayAccess) {
            return getType(((ArrayAccess)e).getArrayName());
        } else if (e instanceof FunctionCall) {
            return getType(((FunctionCall)e).getName());
        } else if (e instanceof Typecast) {
            return new ArrayList(((Typecast)e).getSpecifiers());
        } else if (e instanceof ConditionalExpression) {
            ConditionalExpression ce = (ConditionalExpression)e;
            List specs = getType(ce.getTrueExpression());
            if (specs == null || specs.get(0) == Specifier.VOID) {
                return getType(ce.getFalseExpression());
            } else {
                return specs;
            }
        } else if (e instanceof CommaExpression) {
            List<Traversable> children = e.getChildren();
            return getType((Expression)children.get(children.size() - 1));
        } else {   // default behavior: returns the types of the first child
            List<Traversable> children = e.getChildren();
            for (int i = 0; i < children.size(); i++) {
                Traversable child = children.get(i);
                List child_type = getType((Expression)child);
                if (child_type != null) {
                    return child_type;
                }
            }
            return null;
        }
    }

    // findDeclaration with an arbitrary starting point and a target id.
    private static Declaration
        findDeclaration(Traversable tr, IDExpression id) {
        Traversable t = tr;
        while (t != null && !(t instanceof SymbolTable)) {
            t = t.getParent();
        }
        if (t == null) {
            return null;
        } else {
            return findSymbol((SymbolTable)t, id);
        }
    }

    // Serach for declaration of the identifier
    private static Declaration searchDeclaration(Identifier id) {
	//System.out.println("Searching declaration for: "+ id.getName()); // *debug* *AP*
        Declaration ret = null;
        Traversable parent = id.getParent();
        // Broken IR
        if (parent == null) {
            return null;
        }
        // AccessExpression handling.
        if (parent instanceof AccessExpression &&
            id == ((AccessExpression)parent).getRHS()) {
            List specs = getType(((AccessExpression)parent).getLHS());
            Declaration cdecl = findUserDeclaration(id, specs);
            if (cdecl instanceof ClassDeclaration) {
                ret = ((ClassDeclaration)cdecl).findSymbol(id);
            }
            // __builtin__offsetof handling.
        } else if (parent instanceof OffsetofExpression &&
                   id == ((OffsetofExpression)parent).getExpression()) {
            List specs = ((OffsetofExpression)parent).getSpecifiers();
            Declaration cdecl = findUserDeclaration(id, specs);
            if (cdecl instanceof ClassDeclaration) {
                ret = ((ClassDeclaration)cdecl).findSymbol(id);
            }
        } else {
            ret = id.findDeclaration();
            // This code section only deals with a situation that name conflicts
            // in a scope; e.g.) { int a = b; float b; ... }
            if (ret instanceof VariableDeclaration) {
                Traversable t1 = IRTools.getAncestorOfType(id, Statement.class);
                Traversable t2 = IRTools.getAncestorOfType(ret,Statement.class);
                if (t1 != null && t2 != null
                    && t1.getParent() == t2.getParent()) {
                    List<Traversable> children = t1.getParent().getChildren();
                    if (children.indexOf(t1) < children.indexOf(t2)) {
                        ret = findDeclaration(t1.getParent().getParent(), id);
                    }
                }
            }
        }
        // Prints out warning for undeclared functions/symbols.
        if (ret == null) {
            if (parent instanceof FunctionCall &&
                id == ((FunctionCall)parent).getName()) {
                System.err.print("[WARNING] Function without declaration ");
            } else {
                System.err.print("[WARNING] Undeclared symbol ");
            }
            System.err.println(id + " from " + parent);
        }
        return ret;
    }

    // Find the body of user-defined class declaration
    private static Declaration findUserDeclaration(Traversable tr, List specs) {
        if (specs == null) {
            return null;
        }
        // Find the leading user specifier
        UserSpecifier uspec = null;
        for (int i = 0; i < specs.size(); i++) {
            Object o = specs.get(i);
            if (o instanceof UserSpecifier) {
                uspec = (UserSpecifier)o;
                break;
            }
        }
        if (uspec == null) {
            return null;
        }
        // Find declaration for the user specifier
        Declaration ret = findDeclaration(tr, uspec.getIDExpression());
        // Handles identifier that shares the same name with its type;
        // e.g. typedef struct {} foo; foo foo;
        if (ret instanceof VariableDeclaration &&
            specs == ((VariableDeclaration)ret).getSpecifiers()) {
            Traversable t = IRTools.getAncestorOfType(ret, SymbolTable.class);
            ret = findDeclaration(t.getParent(), uspec.getIDExpression());
        }
        // Keep searching through the chain ( i.e. typedef, etc )
        if (ret instanceof VariableDeclaration) {
            return findUserDeclaration(
                    tr, ((VariableDeclaration)ret).getSpecifiers());
        }
        // Differentiate prototype and actual declaration (forward declaration)
        if (ret instanceof ClassDeclaration && ret.getChildren() == null) {
            IDExpression class_name = ((ClassDeclaration)ret).getName();
            Traversable t = ret.getParent();
            while (t != null) {
                if (t instanceof SymbolTable) {
                    List<Traversable> children = t.getChildren();
                    for (int i = 0; i < children.size(); i++) {
                        Traversable child = children.get(i);
                        if (child instanceof ClassDeclaration) {
                            ClassDeclaration cdecl = (ClassDeclaration)child;
                            if (cdecl.getName().equals(class_name) &&
                                cdecl.getChildren() != null) {
                                ret = cdecl;
                                break;
                            }
                        }
                    }
                }
                t = t.getParent();
            }
        }
        return ret;
    }

    /**
    * Checks if the given symbol is declared as a global variable.
    * @param symbol the symbol of the variable.
    * @return true if the variable is global.
    */
    public static boolean isGlobal(Symbol symbol) {
        if (symbol instanceof PseudoSymbol) {
            symbol = ((PseudoSymbol)symbol).getIRSymbol();
        }
        Traversable tr = (Traversable)symbol;
        while (tr != null && !(tr instanceof SymbolTable)) {
            tr = tr.getParent();
        }
        return (tr instanceof TranslationUnit);
    }

    /**
    * Checks if the given symbol is declared as a local variable.
    * This checking should be made along with other types of checking since
    * the formal parameters are not part of the IR tree.
    * @param symbol the symbol of the variable.
    * @return true if the variable is local.
    */
    public static boolean isLocal(Symbol symbol) {
        if (symbol instanceof PseudoSymbol) {
            symbol = ((PseudoSymbol)symbol).getIRSymbol();
        }
        Traversable tr = (Traversable)symbol;
        while (tr != null && !(tr instanceof SymbolTable)) {
            tr = tr.getParent();
        }
        return (tr != null && !(tr instanceof TranslationUnit));
    }

    /**
    * Checks if the given symbol does not belong to the IR tree.
    * There can be two types of orphan symbol -- 1) formal parameters,
    * 2) other temporary variables not in the tree.
    * @param symbol the symbol of the variable.
    * @return true if the variable is orphan.
    */
    public static boolean isOrphan(Symbol symbol) {
        if (symbol instanceof PseudoSymbol) {
            symbol = ((PseudoSymbol)symbol).getIRSymbol();
        }
        Traversable tr = (Traversable)symbol;
        while (tr != null && !(tr instanceof SymbolTable)) {
            tr = tr.getParent();
        }
        return (tr == null);
    }

    /**
    * Checks if the given symbol is declared as a formal variable.
    * This utility function is using a weak point of the current IR hierarchy;
    * formal parameters are not part of the IR tree but just a satellite data of
    * a procedure object.
    * @param symbol the symbol of the variable.
    * @return true if the variable is formal.
    */
    public static boolean isFormal(Symbol symbol, Procedure proc) {
        if (symbol instanceof PseudoSymbol) {
            symbol = ((PseudoSymbol)symbol).getIRSymbol();
        }
        Traversable t = (Traversable)symbol;
        while (!(t instanceof ProcedureDeclarator || t instanceof SymbolTable)){
            t = t.getParent();
        }
        return (t == proc.getDeclarator() && (t != symbol));
    }

    /**
    * Checks if the given symbol is declared as a formal variable.
    * @param symbol the symbol to be checked.
    * @return true if the symbol represents a formal parameter, false otherwise.
    */
    public static boolean isFormal(Symbol symbol) {
        if (symbol instanceof PseudoSymbol) {
            symbol = ((PseudoSymbol)symbol).getIRSymbol();
        }
        Traversable t = (Traversable)symbol;
        while (!(t instanceof ProcedureDeclarator || t instanceof SymbolTable)){
            t = t.getParent();
        }
        return ((t instanceof ProcedureDeclarator) && (t != symbol));
    }

    /**
    * Returns a list of symbols by taking symbol object out of the expressions
    * in the given list.
    * @param exprs the list of requested expressions.
    * @return the list of symbols that represent each expression.
    */
    public static List<Symbol> exprsToSymbols(List<Expression> exprs) {
        List<Symbol> ret = new ArrayList<Symbol>(exprs.size());
        for (int i = 0; i < exprs.size(); i++) {
            ret.add(getSymbolOf(exprs.get(i)));
        }
        return ret;
    }

    /**
    * Checks if the given symbol is a parameter and a pointer-compatible symbol.
    * Any array declarator is a pointer-compatible symbol.
    * @param param the parameter symbol to be checked.
    * @return true if {@code param} is pointer-compatible.
    */
    public static boolean isPointerParameter(Symbol param) {
        return (isFormal(param) && (isPointer(param) || isArray(param)));
    }

    /**
    * Returns C-native specifiers for the given symbol.
    * @param t the traversable object where search starts from.
    * @param symbol the symbol object of interest.
    * @return the list of resolved specifiers without any user specifiers.
    */
    @SuppressWarnings("unchecked")
    public static List getNativeSpecifiers(Traversable t, Symbol symbol) {
        if (symbol == null) {
            return null;
        }
        List ret = new ArrayList();
        List types = symbol.getTypeSpecifiers();
        for (int i = 0; i < types.size(); i++) {
            Object o = types.get(i);
            if (o instanceof UserSpecifier) {
                List original_types = getNativeSpecifiers(t, (UserSpecifier)o);
                if (original_types == null) {
                    return null;
                }
                ret.addAll(original_types);
            } else {
                ret.add(o);
            }
        }
        ret.addAll(symbol.getArraySpecifiers());
        return ret;
    }

    /**
    * Recursively expands the specified user specifier with its original
    * specifier found in variable declarator that starts with {@code typedef}.
    * @param t the traversable obejct where search starts from.
    * @param uspec the user specifier to be expanded.
    * @return the expanded list of specifiers or null if unsuccessful.
    */
    @SuppressWarnings("unchecked")
    private static List getNativeSpecifiers(Traversable t,
                                            UserSpecifier uspec) {
        List ret = new ArrayList();
        IDExpression id = uspec.getIDExpression();
        SymbolTable symtab = IRTools.getAncestorOfType(t, SymbolTable.class);
        Declaration decl = findSymbol(symtab, id);
        if (decl == null) {
            return null;
        }
        if (decl instanceof ClassDeclaration) {
            ret.add(uspec);
        } else if (decl instanceof Enumeration) {
            ret.add(uspec);
        } else if (decl instanceof VariableDeclaration) {
            VariableDeclaration declaration = (VariableDeclaration)decl;
            Declarator declarator = declaration.getDeclarator(0);
            if (!(declarator instanceof VariableDeclarator)) {
                PrintTools.printlnStatus(1,
                    "[WARNING] cannot extract native types from", declaration);
                return null;
            }
            List specs = ((Symbol)declarator).getTypeSpecifiers();
            if (specs == null || specs.isEmpty() ||
                specs.get(0) != Specifier.TYPEDEF) {
                PrintTools.printlnStatus(1,
                    "[WARNING] cannot extract native types from", declaration);
                return null;
            }
            for (int i = 1; i < specs.size(); i++) {    // skips typedef at 0
                Object o = specs.get(i);
                if (o instanceof UserSpecifier) {
                    List utypes = getNativeSpecifiers(t, (UserSpecifier)o);
                    if (utypes == null) {
                        return null;
                    }
                    ret.addAll(utypes);
                } else {
                    ret.add(o);
                }
            }
        }
        return ret;
    }

    /**
    * Checks if the given symbol is a user-defined struct type. Notice that the
    * first parameter symbol should exist as a traversable object.
    * @param symbol the symbol object to be checked.
    * @param tr the traversable object to be searched.
    */
    public static boolean isStruct(Symbol symbol, Traversable tr) {
        return (getClassDeclaration(symbol, tr) != null);
    }

    /**
    * Returns the user-defined class declaration for the given symbol. Notice
    * that it does not deferentiate "a" and "*a". 
    * @param symbol the symbol object to be checked.
    * @param tr the traversable object to be searched.
    */
    public static ClassDeclaration
        getClassDeclaration(Symbol symbol, Traversable tr) {
        Symbol sym = symbol;
        if (sym instanceof PseudoSymbol) {
            sym = ((PseudoSymbol)sym).getIRSymbol();
        }
        Declaration ret = findUserDeclaration(tr, sym.getTypeSpecifiers());
        if (ret instanceof ClassDeclaration) {
            return (ClassDeclaration)ret;
        } else {
            return null;
        }
    }

    /**
    * Returns an incomplete Identifier whose relevant symbol is not defined.
    * This method is equivalent to the constructor of Identifier with a raw
    * string name, which is now hidden to external world.
    */
    public static Identifier getOrphanID(String name) {
        return new Identifier(name);
    }

    /**
    * Adds symbols to a symbol table and checks for duplicates.
    * @param table The symbol table to add the symbols to.
    * @param decl The declaration of the symbols.
    * symbols in the table.
    */
    public static void addSymbols(SymbolTable table, Declaration decl) {
        Map<IDExpression, Declaration> symbol_table = getTable(table);
        List<IDExpression> ids = decl.getDeclaredIDs();
        int ids_size = ids.size();
        for (int i = 0; i < ids_size; i++) {
            IDExpression id = ids.get(i);
            // Skips dummy identifiers.
            if (id.toString().length() == 0 || id.toString().equals("...")) {
                continue;
            }
            // Overwriting is not allowed except for the case where Procedure
            // declaration overwrites Procedure declarator. This is so because
            // we assign Procedure as the symbol of the relevant Identifier not
            // the procedure declarator. ProcedureDeclarator is used only if
            // there is no procedure body (such as library call).
            if (symbol_table.containsKey(id) && !(decl instanceof Procedure)) {
                continue;
            }
            // Assigns a new identifier if the id represents a variable's name
            // that has a matching declaration/declarator. This
            // table-populating process results in a map from IDExpression
            // objects (NameID/Identifier) to matching declarations. NameID is
            // used for non-variable type IDs such as user-defined type names
            // which cannot have a reference to the Symbol object.
            if (decl instanceof Procedure) {
                symbol_table.put(new Identifier((Symbol)decl), decl);
            } else {
                symbol_table.put(id, decl);
                DFIterator<Traversable> symbol_iter =
                        new DFIterator<Traversable>(decl);
                symbol_iter.pruneOn(NestedDeclarator.class);
                while (symbol_iter.hasNext()) {
                    Traversable t = symbol_iter.next();
                    if (!(t instanceof Symbol)) {
                        continue;
                    }
                    Symbol symbol = (Symbol)t;
                    if (symbol.getSymbolName().equals(id.toString())) {
                        // hash key is not overwritten by put().
                        symbol_table.remove(id);
                        symbol_table.put(new Identifier(symbol), decl);
                        break;
                    }
                }
            }
        }
    }

    /**
    * Searches for a symbol by name in the table.  If the symbol is
    * not in the table, then parent tables will be searched.
    * @param table The initial table to search.
    * @param name The name of the symbol to locate.
    * @return a Declaration if the symbol is found, or null if it is not found.
    *    The Declaration may contain multiple declarators, of which name will
    *    be one, unless the SingleDeclarator pass has been run on the program.
    */
    public static Declaration findSymbol(SymbolTable table, IDExpression name) {
        Declaration ret = null;
        SymbolTable symtab = table;
        while (ret == null && symtab != null) {
            ret = getTable(symtab).get(name);
            symtab = IRTools.getAncestorOfType(symtab, SymbolTable.class);
        }
        return ret;
    }

    /**
    * Returns a list of parent symbol tables.
    */
    protected static List<SymbolTable> getParentTables(Traversable obj) {
        List<SymbolTable> ret = new ArrayList<SymbolTable>();
        Traversable p = obj.getParent();
        while (p != null) {
            if (p instanceof SymbolTable) {
                ret.add((SymbolTable)p);
            }
            p = p.getParent();
        }
        return ret;
    }

    /**
    * Returns a randomly-generated name that is not found in the table.
    * @param table The table to search.
    * @return a unique name.
    */
    public static IDExpression getUnusedID(SymbolTable table) {
        String name = null;
        IDExpression ident = null;
        Random rand = new Random();
        do {
            name = "";
            name += (char)('a' + rand.nextInt(26));
            name += (char)('a' + rand.nextInt(26));
            name += (char)('a' + rand.nextInt(26));
            name += (char)('a' + rand.nextInt(26));
            ident = new NameID(name);
        } while (findSymbol(table, ident) != null);
        return ident;
    }

    /**
    * Removes the symbols declared by the declaration from the symbol table.
    * @param table The table from which to remove the symbols.
    * @param decl The declaration of the symbols.
    */
    protected static void removeSymbols(SymbolTable table, Declaration decl) {
        Map<IDExpression, Declaration> symbol_table = getTable(table);
        List<IDExpression> names = decl.getDeclaredIDs();
        for (int i = 0; i < names.size(); i++) {
            IDExpression id = names.get(i);
            if (symbol_table.remove(id) == null) {
                PrintTools.printlnStatus(0,
                         "[WARNING] Attempt to remove a non-existing symbol",
                         "\"", id, "\"", "in", table.getClass());
            }
        }
    }

    /**
    * Removes the specified symbol from the symbol table object. This methods
    * consistently modifies the look-up table, and the IR. No action occurs if
    * the specified symbol is not found in the symbol table object.
    * @param table the relevant symbol table object.
    * @param symbol the symbol object to be removed.
    * @throws IllegalArgumentException if <b>symbol</b> is referenced within
    * <b>table</b>.
    */
/* TODO: will finish later - remove from release branch
    protected static void removeSymbol(SymbolTable table, Symbol symbol) {
        Set<Symbol> symbols = table.getSymbols();
        if (!symbols.contains(symbol)) {
            return;
        }
        Set<Symbol> accessed_symbols = getAccessedSymbols(table);
        if (accessed_symbols.contains(symbol)) {
            throw new IllegalArgumentException(
                    "Cannot remove a referenced symbol.");
        }
        if (symbol instanceof Declaration) {
            table.removeDeclaration((Declaration)symbol);
        } else if
    }
*/

    /**
    * Returns a new identifier derived from the given identifier.
    * This method internally calls {@link #getTemp(Identifier, String)}.
    * @param id the identifier from which type and scope are derived.
    * @return the new identifier.
    */
    public static Identifier getTemp(Identifier id) {
        return getTemp(id, id.getName());
    }

    /**
    * Returns a new identifier derived from the given IR object and identifier.
    * This method internally calls {@link #getTemp(Traversable, List, String)}.
    * @param where the IR object from which scope is derived.
    * @param id the identifier from which type is derived.
    * @return the new identifier.
    */
    public static Identifier getTemp(Traversable where, Identifier id) {
        return getTemp(where, id.getSymbol().getTypeSpecifiers(), id.getName());
    }

    /**
    * Returns a new identifier derived from the given identifier and name.
    * This method internally calls {@link #getTemp(Traversable, List, String)}.
    * @param id the identifier from which scope is derived.
    * @param name the string from which name is derived.
    * @return the new identifier.
    */
    public static Identifier getTemp(Identifier id, String name) {
        return getTemp(id, id.getSymbol().getTypeSpecifiers(), name);
    }

    /**
    * Returns a new identifier derived from the given IR object, type, and name.
    * This method internally calls {@link #getTemp(Traversable, List, String)}.
    * @param where the IR object from which scope is derived.
    * @param spec the type specifier.
    * @param name the string from which name is derived.
    * @return the new identifier.
    */
    public static Identifier getTemp(Traversable where,
                                     Specifier spec, String name) {
        List<Specifier> specs = new ArrayList<Specifier>(1);
        specs.add(spec);
        return getTemp(where, specs, name);
    }

    /**
    * Returns a new identifier derived from the given IR object, type list, and
    * name. This method internally calls
    * {@link #getArrayTemp(Traversable, List, List, String)}.
    * @param where the IR object from which scope is derived.
    * @param specs the type specifiers.
    * @param name the string from which name is derived.
    * @return the new identifier.
    */
    public static Identifier getTemp(Traversable where,
                                     List specs, String name) {
        return getArrayTemp(where, specs, (List)null, name);
    }

    /**
    * Returns a new identifier derived from the given IR object, type list,
    * array specifier and name. This method internally calls
    * {@link #getArrayTemp(Traversable, List, List, String)}.
    * @param where the IR object from which scope is derived.
    * @param specs the type specifiers.
    * @param aspec the array specifier.
    * @param name the string from which name is derived.
    * @return the new identifier.
    */
    public static Identifier getArrayTemp(Traversable where, List specs,
                                          ArraySpecifier aspec,
                                          String name) {
        List<Specifier> aspecs = new ArrayList<Specifier>(1);
        aspecs.add(aspec);
        return getArrayTemp(where, specs, aspecs, name);
    }

    /**
    * Returns a new identifier derived from the given IR object, type list,
    * array specifiers and name. If {@code specs} contains any pointer
    * specifiers, they are automatically separated and inserted into the list
    * of specifiers that belong to the new {@code VariableDeclarator} object.
    * @param where the IR object from which scope is derived.
    * @param specs the type specifiers.
    * @param aspecs the array specifier.
    * @param name the string from which name is derived.
    * @return the new identifier.
    */
    @SuppressWarnings("unchecked")
    public static Identifier getArrayTemp(Traversable where, List specs,
                                          List aspecs, String name) {
        SymbolTable st;
        if (where instanceof SymbolTable) {
            st = (SymbolTable)where;
        } else {
            st = IRTools.getAncestorOfType(where, SymbolTable.class);
        }
        if (st instanceof Loop) {
            st = IRTools.getAncestorOfType(st, SymbolTable.class);
        }
        String header = (name == null) ? "_temp_" : name + "_";
        NameID id = null;
        for (int trailer = 0; id == null; ++trailer) {
            NameID newid = new NameID(header + trailer);
            if (findSymbol(st, newid) == null) {
                id = newid;
            }
        }
        // Separate declarator/declaration specifiers.
        List declaration_specs = new ArrayList(specs.size());
        List declarator_specs = new ArrayList(specs.size());
        for (int i = 0; i < specs.size(); i++) {
            Object spec = specs.get(i);
            if (spec instanceof PointerSpecifier) {
                declarator_specs.add(spec);
            } else {
                declaration_specs.add(spec);
            }
        }
        VariableDeclarator decl = null;
        if (declarator_specs.isEmpty()) {
            if (aspecs == null || aspecs.isEmpty()) {
                decl = new VariableDeclarator(id);
            } else {
                decl = new VariableDeclarator(id, aspecs);
            }
        } else if (aspecs == null || aspecs.isEmpty()) {
            decl = new VariableDeclarator(declarator_specs, id);
        } else {
            decl = new VariableDeclarator(declarator_specs, id, aspecs);
        }
        Declaration decls = new VariableDeclaration(declaration_specs, decl);
        st.addDeclaration(decls);
        return new Identifier(decl);
    }

    /**
    * Returns a new, pointer-type identifier derived from the given IR object.
    * @param where the IR object from which scope is derived.
    * @param refID the identifier from which type and name are derived.
    * @return the new pointer-type identifier.
    */
    public static Identifier getPointerTemp(Traversable where,
                                            Identifier refID) {
        List<Specifier> pspecs = new LinkedList<Specifier>();
        pspecs.add(PointerSpecifier.UNQUALIFIED);
        return getPointerTemp(where, refID.getSymbol().getTypeSpecifiers(),
                              pspecs, refID.getName());
    }

    /**
    * Returns a new, pointer-type identifier derived from the given IR object.
    * @param where the IR object from which scope is derived.
    * @param specs the type specifiers.
    * @param name the string from which name is derived.
    * @return the new pointer-type identifier.
    */
    public static Identifier getPointerTemp(Traversable where, List specs,
                                            String name) {
        List<Specifier> pspecs = new LinkedList<Specifier>();
        pspecs.add(PointerSpecifier.UNQUALIFIED);
        return getPointerTemp(where, specs, pspecs, name);
    }

    /**
    * Returns a new, pointer-type identifier derived from the given IR object.
    * @param where the IR object from which scope is derived.
    * @param specs the type specifiers.
    * @param pspecs the pointer-type specifiers.
    * @param name the string from which name is derived.
    * @return the new pointer-type identifier.
    */
    public static Identifier getPointerTemp(Traversable where, List specs,
                                            List pspecs, String name) {
        Traversable t = where;
        while (!(t instanceof SymbolTable)) {
            t = t.getParent();
        }
        // Traverse to the parent of a loop statement
        if (t instanceof Loop) {
            t = t.getParent();
            while (!(t instanceof SymbolTable)) {
                t = t.getParent();
            }
        }
        SymbolTable st = (SymbolTable)t;
        String header = (name == null) ? "_temp_" : name + "_";
        NameID id = null;
        for (int trailer = 0; id == null; ++trailer) {
            NameID newid = new NameID(header + trailer);
            if (findSymbol(st, newid) == null) {
                id = newid;
            }
        }
        VariableDeclarator decl = new VariableDeclarator(pspecs, id);
        Declaration decls = new VariableDeclaration(specs, decl);
        st.addDeclaration(decls);
        return new Identifier(decl);
    }

    /**
    * Returns a new name id with the given name suggestion and scope.
    * @param name the name suggestion given by user.
    * @param tr the traversable object which scope is derived from.
    */
    public static NameID getNewName(String name, Traversable tr) {
        SymbolTable symtab = IRTools.getAncestorOfType(tr, SymbolTable.class);
        String header = (name == null) ? "temp" : name;
        NameID ret = new NameID(header);
        int suffix = 0;
        while (findSymbol(symtab, ret) != null) {
            ret = new NameID(header + (suffix++));
        }
        return ret;
    }

    /**
    * Returns the set of Symbol objects contained in the given SymbolTable
    * object.
    * @param st the symbol table being searched.
    * @return the set of symbols.
    */
    public static Set<Symbol> getSymbols(SymbolTable st) {
        Set<Symbol> ret = new LinkedHashSet<Symbol>();
        if (st == null) {
            return ret;
        }
        for (IDExpression key : getTable(st).keySet()) {
            if (key instanceof Identifier) {
                Symbol symbol = ((Identifier)key).getSymbol();
                if (symbol != null) {
                    ret.add(symbol);
                }
            }
        }
        return ret;
    }

    /**
    * Returns the set of Symbol objects contained in the given SymbolTable
    * object excluding Procedures.
    * @param st the symbol table being searched.
    * @return the set of symbols.
    */
    public static Set<Symbol> getVariableSymbols(SymbolTable st) {
        Set<Symbol> ret = getSymbols(st);
        Iterator<Symbol> iter = ret.iterator();
        while (iter.hasNext()) {
            Symbol symbol = iter.next();
            if (symbol instanceof Procedure ||
                symbol instanceof ProcedureDeclarator) {
                iter.remove();
            }
        }
        return ret;
    }

    /**
    * Returns the set of Symbol objects that are global variables 
    * of the File scope 
    */
    public static Set<Symbol> getGlobalSymbols(Traversable t) {
        while (true) {
            if (t instanceof TranslationUnit) {
                break;
            }
            t = t.getParent();
        }
        TranslationUnit t_unit = (TranslationUnit)t;
        return getVariableSymbols(t_unit);
    }

    /**
    * Returns the set of Symbol objects that are formal parameters of 
    * the given Procedure
    */
    public static Set<Symbol> getParameterSymbols(Procedure proc) {
        Set<Symbol> ret = new LinkedHashSet<Symbol>();
        DFIterator<Traversable> iter =
            new DFIterator<Traversable>(proc.getDeclarator());
        iter.pruneOn(NestedDeclarator.class);
        iter.next();            // skip procedure declarator itself.
        while (iter.hasNext()) {
            Traversable t = iter.next();
            if (t instanceof Symbol) {
                ret.add((Symbol)t);
            }
        }
        return ret;
    }

    public static Set<Symbol> getSideEffectSymbols(FunctionCall fc) {
        Set<Symbol> side_effect_set = new HashSet<Symbol>();
        // set of GlobalVariable Symbols that are accessed within a Procedure
        Procedure proc = fc.getProcedure();
        // we assume that there is no global variable access within a procedure
        // if a procedure body is not available for a compiler
        // example: system calls
        if (proc != null) {
            Set<Symbol> global_variables = new HashSet<Symbol>();
            Set<Symbol> accessed_symbols =
                getAccessedSymbols(proc.getBody());
            for (Symbol var : accessed_symbols) {
                if (isGlobal(var, proc)) {
                    global_variables.add(var);
                }
            }
            if (!global_variables.isEmpty()) {
                side_effect_set.addAll(global_variables);
            }
        }
        // find the set of actual parameter Symbols of each function call
        List<Expression> arguments = fc.getArguments();
        HashSet<Symbol> parameters = new HashSet<Symbol>();
        for (Expression e : arguments) {
            parameters.addAll(getAccessedSymbols(e));
        }
        if (!parameters.isEmpty()) {
            side_effect_set.addAll(parameters);
        }
        return side_effect_set;
    }

    /**
    * Returns the set of symbols accessed in the traversable object.
    * @param t the traversable object.
    * @return the set of symbols.
    */
    public static Set<Symbol> getAccessedSymbols(Traversable t) {
        Set<Symbol> ret = new HashSet<Symbol>();
        if (t == null) {
            return ret;
        }
        DFIterator<Identifier> iter =
            new DFIterator<Identifier>(t, Identifier.class);
        while (iter.hasNext()) {
            Symbol symbol = iter.next().getSymbol();
            if (symbol != null) {
                ret.add(symbol);
            }
        }
        return ret;
    }

    /**
    * Returns the set of symbols declared within the specified traversable
    * object.
    * @param t the traversable object.
    * @return the set of such symbols.
    */
    public static Set<Symbol> getLocalSymbols(Traversable t) {
        Set<Symbol> ret = new LinkedHashSet<Symbol>();
        if (t == null) {
            return ret;
        }
        DFIterator<SymbolTable> iter =
                new DFIterator<SymbolTable>(t, SymbolTable.class);
        while (iter.hasNext()) {
            ret.addAll(getSymbols(iter.next()));
        }
        return ret;
    }

    /**
    * Returns the symbol object having the specified string name.
    * @param name the name to be searched for.
    * @param tr the IR location where searching starts.
    * @return the symbol object.
    */
    public static Symbol getSymbolOfName(String name, Traversable tr) {
        Symbol ret = null;
        Traversable t = tr;
        while (ret == null && t != null) {
            if (t instanceof SymbolTable) {
                for (Symbol symbol : getSymbols((SymbolTable)t)) {
                    if (name.equals(symbol.getSymbolName())) {
                        ret = symbol;
                        break;
                    }
                }
            }
            t = t.getParent();
        }
        return ret;
    }

    /**
    * Returns the symbol of the expression if it represents an lvalue.
    * @param e the input expression.
    * @return the corresponding symbol object.
    */
    // The following symbol is returned for each expression types.
    // Identifier         : its symbol.
    // ArrayAccess        : base name's symbol.
    // AccessExpression   : access symbol (list of symbols).
    // Pointer Dereference: the first symbol found in the expression tree.
    public static Symbol getSymbolOf(Expression e) {
        if (e instanceof Identifier) {
            return ((Identifier)e).getSymbol();
        } else if (e instanceof ArrayAccess) {
            return getSymbolOf(((ArrayAccess)e).getArrayName());
        } else if (e instanceof AccessExpression) {
            return new AccessSymbol((AccessExpression)e);
        } else if (e instanceof UnaryExpression) {
            return getSymbolOf(((UnaryExpression)e).getExpression());
/*
            UnaryExpression ue = (UnaryExpression)e;
            if (ue.getOperator() == UnaryOperator.DEREFERENCE) {
                DFIterator<Identifier> iter = new DFIterator<Identifier>(
                        ue.getExpression(), Identifier.class);
                if (iter.hasNext()) {
                    return iter.next().getSymbol();
                }
            }
*/
        } else if (e instanceof Typecast) {
            return getSymbolOf(((Typecast)e).getExpression());
        /**
         * TAN: It is quite tricky to find symbol for NameID type.
         * Firstly we have to find VariableDeclaration, then VariableDeclarator,
         * and then cast it to Symbol
         */
        } else if (e instanceof NameID) {
        	VariableDeclaration eDeclion = (VariableDeclaration)((IDExpression)e).findDeclaration();
        	VariableDeclarator eDeclor = (VariableDeclarator)eDeclion.getDeclarator(0);
        	return (Symbol)eDeclor;
        }
        return null;
    }

    /**
    * Checks if the symbol is a global variable to the procedure containing the
    * given traversable object.
    * @param symbol The symbol object
    * @param t The traversable object
    * @return true if it is global, false otherwise
    */
    public static boolean isGlobal(Symbol symbol, Traversable t) {
        SymbolTable symtab = IRTools.getAncestorOfType(t, Procedure.class);
        if (t == null) {
            return true;        // conservative decision if a bad thing happens.
        }
        while ((symtab = IRTools.getAncestorOfType(symtab, SymbolTable.class))
               != null) {
            if (symtab.containsSymbol(symbol)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Checks if the symbol is a scalar variable.
    * @param symbol The symbol
    * @return true if it is a scalar variable, false otherwise
    */
    public static boolean isScalar(Symbol symbol) {
        if (symbol == null) {
            return false;
        }
        List specs = symbol.getArraySpecifiers();
        return (specs == null || specs.isEmpty());
    }

    /**
    * Checks if the symbol is an array variable.
    * @param symbol The symbol
    * @return true if it is an array variable, false otherwise
    */
    public static boolean isArray(Symbol symbol) {
        if (symbol == null) {
            return false;
        }
        List specs = symbol.getArraySpecifiers();
        return (specs != null && !specs.isEmpty());
    }

    /**
    * Checks if the symbol is a pointer type variable.
    * @param symbol The symbol
    * @return true if it is a pointer type variable, false otherwise
    */
    public static boolean isPointer(Symbol symbol) {
        if (symbol == null) {
            return false;
        }
        List specs = symbol.getTypeSpecifiers();
        if (specs == null) {
            return false;
        }
        for (int i = 0; i < specs.size(); i++) {
            Object o = specs.get(i);
            if (o instanceof PointerSpecifier) {
                return true;
            }
        }
        return false;
    }

    /**
    * Checks if the symbol is a pointer type variable. The input expression
    * should represent a variable. Otherwise it will return true.
    * @param e the expression to be tested.
    */
    public static boolean isPointer(Expression e) {
        List spec = getExpressionType(e);
        if (spec == null || spec.isEmpty() ||
            spec.get(spec.size() - 1) instanceof PointerSpecifier) {
            return true;
        } else {
            return false;
        }
    }

    // For use with isInteger()
    private static final Set<Specifier> int_types;

    static {
        int_types = new HashSet<Specifier>();
        int_types.add(Specifier.INT);
        int_types.add(Specifier.LONG);
        int_types.add(Specifier.SIGNED);
        int_types.add(Specifier.UNSIGNED);
    }

    /**
    * Checks if the symbol is an interger type variable.
    * @param symbol the symbol.
    * @return true if it is an integer type variable, false otherwise.
    */
    public static boolean isInteger(Symbol symbol) {
        return (symbol != null && isInteger(symbol.getTypeSpecifiers()));
    }

    public static boolean isInteger(List specifiers) {
        if (specifiers == null) {
            return false;
        }
        boolean ret = false;
        for (int i = 0; i < specifiers.size(); i++) {
            Object o = specifiers.get(i);
            if (o == Specifier.CHAR || o instanceof PointerSpecifier) {
                return false;
            }
            ret |= int_types.contains(o);
        }
        return ret;
    }

    /**
    * Returns the exact type of the given expression, tracking all specifiers
    * relevant to the expression. For example, the returned list contains the
    * type specifiers and array specifiers for an identifier declared as an
    * array type.
    * @param e the expression to be examined.
    * @return the type of the expression.
    */
    @SuppressWarnings("unchecked")
    public static List<Specifier> getExactExpressionType(Expression e) {
        List<Specifier> ret = null;
        if (e instanceof Identifier) {
            Symbol symbol = ((Identifier)e).getSymbol();
            if (symbol != null) {
                ret = new ArrayList<Specifier>();
                ret.addAll(symbol.getTypeSpecifiers());
                ret.addAll(symbol.getArraySpecifiers());
                if (symbol instanceof NestedDeclarator) {
                    Declarator nested =
                            ((NestedDeclarator)symbol).getDeclarator();
                    if (nested instanceof VariableDeclarator) {
                        ret.addAll(
                                ((VariableDeclarator)nested).getSpecifiers());
                    }
                }
            }
        } else if (e instanceof ArrayAccess) {
            ret = getExactExpressionType(((ArrayAccess)e).getArrayName());
            if (ret != null) {
                for (int i = 0; i < ((ArrayAccess)e).getNumIndices();) {
                    Specifier spec = ret.remove(ret.size() - 1);
                    if (spec instanceof ArraySpecifier) {
                        i += ((ArraySpecifier)spec).getNumDimensions();
                    } else if (spec instanceof PointerSpecifier) {
                        i++;
                    } else {
                        ret = null;
                        break;
                    }
                }
            }
        } else if (e instanceof AccessExpression) {
            ret = getExactExpressionType(((AccessExpression)e).getRHS());
        } else if (e instanceof AssignmentExpression) {
            ret = getExactExpressionType(((AssignmentExpression)e).getLHS());
        } else if (e instanceof CommaExpression) {
            List<Traversable> children = e.getChildren();
            ret = getExactExpressionType(
                    (Expression)children.get(children.size() - 1));
        } else if (e instanceof ConditionalExpression) {
            ret = getExactExpressionType(
                    ((ConditionalExpression)e).getTrueExpression());
        } else if (e instanceof FunctionCall) {
            ret = getExactExpressionType(((FunctionCall)e).getName());
        } else if (e instanceof Literal) {
            ret = new ArrayList<Specifier>(2);
            if (e instanceof IntegerLiteral) {
                ret.add(Specifier.LONG);
            } else if (e instanceof BooleanLiteral) {
                ret.add(Specifier.BOOL);
            } else if (e instanceof CharLiteral) {
                ret.add(Specifier.CHAR);
            } else if (e instanceof StringLiteral) {
                ret.add(Specifier.CHAR);
                ret.add(PointerSpecifier.UNQUALIFIED);
            } else if (e instanceof FloatLiteral) {
                ret.add(Specifier.DOUBLE);
            }
        } else if (e instanceof Typecast) {
            ret = new ArrayList<Specifier>();
            List specs = ((Typecast)e).getSpecifiers();
            for (int i = 0; i < specs.size(); i++) {
                Object spec = specs.get(i);
                if (spec instanceof Specifier) {
                    ret.add((Specifier)spec);
                } else if (spec instanceof Declarator) {
                    ret.addAll(((Declarator)spec).getSpecifiers());
                } else {
                    ret = null;
                    break;
                }
            }
        } else if (e instanceof UnaryExpression) {
            ret = getExactExpressionType(((UnaryExpression)e).getExpression());
            if (ret != null) {
                UnaryOperator op = ((UnaryExpression)e).getOperator();
                if (op == UnaryOperator.ADDRESS_OF) {
                    ret.add(PointerSpecifier.UNQUALIFIED);
                } else if (op == UnaryOperator.DEREFERENCE) {
                    if (ret.get(ret.size() - 1) instanceof PointerSpecifier) {
                        ret.remove(ret.size() - 1);
                    } else {
                        ret = null;
                    }
                }
            }
        } else if (e instanceof BinaryExpression) {
            BinaryExpression be = (BinaryExpression)e;
            if (be.getOperator().isCompare() || be.getOperator().isLogical()) {
                ret = new ArrayList<Specifier>(1);
                ret.add(Specifier.LONG);
            } else if (be.getLHS() instanceof Literal) {
                ret = getExactExpressionType(be.getRHS());
            } else {
                ret = getExactExpressionType(be.getLHS());
            }
        } else if (e instanceof VaArgExpression) {
            ret = ((VaArgExpression)e).getSpecifiers();
        }
        if (ret == null) {
            PrintTools.printlnStatus(1, "[WARNING] Unknown expression type:",e);
        }
        return ret;
    }

    /**
    * Returns a list of specifiers of the given expression.
    * @param e the given expression.
    * @return the list of specifiers.
    */
    @SuppressWarnings("unchecked")
    public static List getExpressionType(Expression e) {
        if (e instanceof Identifier) {
            Symbol var = ((Identifier)e).getSymbol();
            if (var != null) {
                return var.getTypeSpecifiers();
            }
        } else if (e instanceof ArrayAccess) {
            ArrayAccess aa = (ArrayAccess)e;
            List ret = getExpressionType(aa.getArrayName());
            if (ret != null && !ret.isEmpty()) {
                LinkedList ret0 = new LinkedList(ret);
                for (int i = 0; i < aa.getNumIndices(); ++i) {
                    if (ret0.getLast() instanceof PointerSpecifier) {
                        ret0.removeLast();
                    }
                }
                return ret0;
            }
        } else if (e instanceof AccessExpression) {
            //Symbol var = ((AccessExpression)e).getSymbol();
            //if ( var != null )
            Symbol var = new AccessSymbol((AccessExpression)e);
            return var.getTypeSpecifiers();
        } else if (e instanceof AssignmentExpression) {
            return getExpressionType(((AssignmentExpression)e).getLHS());
        } else if (e instanceof CommaExpression) {
            List<Traversable> children = e.getChildren();
            return getExpressionType(
                    (Expression)children.get(children.size() - 1));
        } else if (e instanceof ConditionalExpression) {
            return getExpressionType(
                    ((ConditionalExpression)e).getTrueExpression());
        } else if (e instanceof FunctionCall) {
            Expression fc_name = ((FunctionCall)e).getName();
            if (fc_name instanceof Identifier) {
                Symbol fc_var = ((Identifier)fc_name).getSymbol();
                if (fc_var != null) {
                    return fc_var.getTypeSpecifiers();
                }
            }
        } else if (e instanceof IntegerLiteral) {
            List ret = new ArrayList(1);
            ret.add(Specifier.LONG);
            return ret;
        } else if (e instanceof BooleanLiteral) {
            List ret = new ArrayList(1);
            ret.add(Specifier.BOOL);
            return ret;
        } else if (e instanceof CharLiteral) {
            List ret = new ArrayList(1);
            ret.add(Specifier.CHAR);
            return ret;
        } else if (e instanceof StringLiteral) {
            List ret = new ArrayList(1);
            ret.add(Specifier.CHAR);
            ret.add(PointerSpecifier.UNQUALIFIED);
            return ret;
        } else if (e instanceof FloatLiteral) {
            List ret = new ArrayList(1);
            ret.add(Specifier.DOUBLE);
            return ret;
        } else if (e instanceof Typecast) {
            List ret = new ArrayList();
            List specs = ((Typecast)e).getSpecifiers();
            for (int i = 0; i < specs.size(); i++) {
                Object spec = specs.get(i);
                if (spec instanceof Specifier) {
                    ret.add(spec);
                } else if (spec instanceof Declarator) {
                    ret.addAll(((Declarator)spec).getSpecifiers());
                }
            }
            return ret;
        } else if (e instanceof UnaryExpression) {
            UnaryExpression ue = (UnaryExpression)e;
            UnaryOperator op = ue.getOperator();
            List ret = getExpressionType(ue.getExpression());
            if (ret != null) {
                LinkedList ret0 = new LinkedList(ret);
                if (op == UnaryOperator.ADDRESS_OF) {
                    ret0.addLast(PointerSpecifier.UNQUALIFIED);
                } else if (op == UnaryOperator.DEREFERENCE) {
                    ret0.removeLast();
                }
                return ret0;
            }
        } else if (e instanceof BinaryExpression) {
            BinaryExpression be = (BinaryExpression)e;
            BinaryOperator op = be.getOperator();
            if (op.isCompare() || op.isLogical()) {
                List ret = new ArrayList(1);
                ret.add(Specifier.LONG);
                return ret;
            } else {
                return getExpressionType(be.getLHS());
            }
        } else if (e instanceof VaArgExpression) {
            return ((VaArgExpression)e).getSpecifiers();
        }
        PrintTools.printlnStatus(1, "[WARNING] Unknown expression type: ", e);
        return null;
    }

    /**
    * Searches for a symbol by String sname in the table. If the symbol is
    * not in the table, then parent tables will be searched breadth-first.
    * If multiple symbols have the same String name, the first one found 
    * during the search will be returned.
    * @param table The initial table to search.
    * @param sname The String name of the symbol (Symbol.getSymbolName()) to
    * locate.
    * @return a Declaration if the symbol is found, or null if it is not found.
    *    The Declaration may contain multiple declarators, of which sname will
    *    be one, unless the SingleDeclarator pass has been run on the program.
    */
    public static Declaration findSymbol(SymbolTable table, String sname) {
        return findSymbol(table, new NameID(sname));
    }

    /**
    * Returns a list of specifiers of the expression.
    */
    @SuppressWarnings("unchecked")
    public static LinkedList getVariableType(Expression e) {
        LinkedList ret = new LinkedList();
        if (e instanceof Identifier) {
            Symbol var = ((Identifier)e).getSymbol();
            if (var != null) {
                ret.addAll(var.getTypeSpecifiers());
            }
        } else if (e instanceof ArrayAccess) {
            ArrayAccess aa = (ArrayAccess)e;
            ret = getVariableType(aa.getArrayName());
            for (int i = 0; i < aa.getNumIndices(); ++i) {
                if (ret.getLast() instanceof PointerSpecifier) {
                    ret.removeLast();
                }
            }
        } else if (e instanceof AccessExpression) {
            //Symbol var = ((AccessExpression)e).getSymbol();
            Symbol var = new AccessSymbol((AccessExpression)e);
            //if ( var != null )
            ret.addAll(var.getTypeSpecifiers());
        } else if (e instanceof UnaryExpression) {
            UnaryExpression ue = (UnaryExpression)e;
            if (ue.getOperator() == UnaryOperator.DEREFERENCE) {
                ret = getVariableType(ue.getExpression());
                if (ret.getLast() instanceof PointerSpecifier) {
                    ret.removeLast();
                } else {
                    ret.clear();
                }
            }
        }
        return ret;
    }

    /**
    * Returns the look-up table that is internal to the given symbol table
    * object.
    * Exposing the table access to public may incur inconsistent snapshot of any
    * symbol table object and its look-up table. Only read-only access will be
    * provided through <b>SymbolTable</b> interface.
    * @param symtab the given symbol table object.
    * @return the internal look-up table, null if any exceptions occur.
    */
    protected static Map<IDExpression, Declaration>
            getTable(SymbolTable symtab) {
        // This looks ugly but inevitable since interface cannot have a
        // protected method and we do not want to expose this method to public.
        if (symtab instanceof ClassDeclaration) {
            return ((ClassDeclaration)symtab).getTable();
        } else if (symtab instanceof CompoundStatement) {
            return ((CompoundStatement)symtab).getTable();
        } else if (symtab instanceof DoLoop) {
            return ((DoLoop)symtab).getTable();
        } else if (symtab instanceof ExceptionHandler) {
            return ((ExceptionHandler)symtab).getTable();
        } else if (symtab instanceof ForLoop) {
            return ((ForLoop)symtab).getTable();
        } else if (symtab instanceof LinkageSpecification) {
            return ((LinkageSpecification)symtab).getTable();
        } else if (symtab instanceof Namespace) {
            return ((Namespace)symtab).getTable();
        } else if (symtab instanceof Procedure) {
            return ((Procedure)symtab).getTable();
        } else if (symtab instanceof TranslationUnit) {
            return ((TranslationUnit)symtab).getTable();
        } else if (symtab instanceof WhileLoop) {
            return ((WhileLoop)symtab).getTable();
        } else {
	    System.out.println("symtab type: "+symtab.getClass().toString()); // *debug* *AP*
            throw new InternalError("[ERROR] Unknown SymbolTable type.");
        }
    }

    /**
    * Updates the symbol links in every Identifier object within the specified
    * symbol table object. This operation is necessary when a symbol table
    * object is cloned (or similar case).
    */
    protected static void relinkSymbols(SymbolTable symtab) {
        // Process inner symbol tables first.
        DFIterator<SymbolTable> iter =
                new DFIterator<SymbolTable>(symtab, SymbolTable.class);
        iter.next();
        iter.pruneOn(SymbolTable.class);
        while (iter.hasNext()) {
            relinkSymbols(iter.next());
        }
        // Process the topmost symbol table.
        Set<Symbol> valid_symbols = symtab.getSymbols();
        DFIterator<Identifier> id_iter =
                new DFIterator<Identifier>(symtab, Identifier.class);
        while (id_iter.hasNext()) {
            Identifier id = id_iter.next();
            Traversable parent = id.getParent();
            // Member of struct should be intact.
            if (!(parent instanceof AccessExpression &&
                        ((AccessExpression)parent).getRHS() == id)) {
                for (Symbol valid_symbol : valid_symbols) {
                    if (id.toString().equals(valid_symbol.getSymbolName())) {
                        id.setSymbol(valid_symbol);
                    }
                }
            }
        }
    }

    /**
    * Renames the specified symbol within the given symbol table object.
    * @param symbol the symbol object to be renamed.
    * @param name the new name of the specified symbol.
    * @param symtab the symbol table object that contains {@code symbol}.
    */
    public static void
     setSymbolName(Symbol symbol, String name, SymbolTable symtab) {
        if (symbol == null || symbol instanceof PseudoSymbol || symtab == null){
            throw new UnsupportedOperationException(
                    "Renaming is supported only for a valid Cetus IR.");
        }
        Declaration entry =
            IRTools.getAncestorOfType((Traversable)symbol, Declaration.class);
        // Remove and update look-up table key/entry while renaming the symbol.
        removeSymbols(symtab, entry);
        if (symbol instanceof Declarator) {
            ((Declarator)symbol).setDirectDeclarator(new NameID(name));
        } else if (symbol instanceof Procedure) {
            ((Procedure)symbol).getDeclarator().setDirectDeclarator(
                    new NameID(name));
        } else {
            throw new UnsupportedOperationException("Unknown symbol type.");
        }
        addSymbols(symtab, entry);
    }

    /**
    * Checks if the specified symbol object contains the given specifier. This
    * functionality can also be provided through
    * {@link Symbol#getTypeSpecifiers} or {@link Symbol#getArraySpecifiers} but
    * they consume extra memory for the returned list.
    * @param symbol the symbol to be checked.
    * @param spec the specifier to be searched for.
    * @return true if a matching specifier is found.
    */
    public static boolean containsSpecifier(Symbol symbol, Specifier spec) {
        if (symbol instanceof VariableDeclarator ||
            symbol instanceof ProcedureDeclarator) {
            Declaration decln = symbol.getDeclaration();
            if (decln instanceof VariableDeclaration &&
                ((VariableDeclaration)decln).getSpecifiers().contains(spec)) {
                return true;
            }
            Declarator declr = (Declarator)symbol;
            if (declr.leading_specs.contains(spec) ||
                declr.trailing_specs.contains(spec)) {
                return true;
            }
        } else if (symbol instanceof Procedure) {
            Procedure proc = (Procedure)symbol;
            if (proc.getSpecifiers().contains(spec)) {
                return true;
            }
            if (proc.getDeclarator().leading_specs.contains(spec) ||
                proc.getDeclarator().trailing_specs.contains(spec)) {
                return true;
            }
        } else if (symbol instanceof NestedDeclarator) {
            Declaration decln = symbol.getDeclaration();
            if (decln instanceof VariableDeclaration ||
                ((VariableDeclaration)decln).getSpecifiers().contains(spec)) {
                return true;
            }
            Declarator declr = (Declarator)symbol;
            if (declr.leading_specs.contains(spec) ||
                declr.trailing_specs.contains(spec)) {
                return true;
            }
            declr = ((NestedDeclarator)declr).getDeclarator();
            if (declr.leading_specs.contains(spec) ||
                declr.trailing_specs.contains(spec)) {
                return true;
            }
        }
        return false;
    }

}

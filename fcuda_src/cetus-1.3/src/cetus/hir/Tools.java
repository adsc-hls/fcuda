package cetus.hir;

import cetus.analysis.RangeDomain;
import cetus.analysis.Section;

import java.io.PrintWriter;
import java.util.*;

/**
* Collection of static tools that are useful in writing passes. The original
* set of tools since v1.1 has been separated into four different classes -
* SymbolTools, PrintTools, IRTools, and Tools:
* <p>
* <b>SymbolTools</b>: symbol access and management.<br>
* <b>PrintTools</b> : printing utilities.<br>
* <b>IRTools</b>    : Cetus IR tools for search and manipulation.<br>
* <b>Tools</b>      : general tools not specific to Cetus.<br>
* <p>
* Accessing the old utility methods through Tools class is still possible but
* it is deprecated As of release 1.2; use the new tools in
* {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
* {@link SymbolTools}.
*/
public final class Tools {

    private Tools() {
    }

    /////////////////////////////////////////////////////////////////////////// 
    // SymbolTools
    /////////////////////////////////////////////////////////////////////////// 

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void addSymbols(SymbolTable table,
                                                    Declaration decl) {
        SymbolTools.addSymbols(table, decl);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Declaration findSymbol(SymbolTable table,
                                                     IDExpression name) {
        return SymbolTools.findSymbol(table, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Declaration findSymbol(SymbolTable table,
                                                     String name) {
        return SymbolTools.findSymbol(table, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated protected static List<SymbolTable>
        getParentTables(Traversable obj) {
        return SymbolTools.getParentTables(obj);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static IDExpression getUnusedID(SymbolTable table) {
        return SymbolTools.getUnusedID(table);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated protected static void removeSymbols(SymbolTable table,
                                                    Declaration decl) {
        SymbolTools.removeSymbols(table, decl);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getTemp(Identifier id) {
        return SymbolTools.getTemp(id);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getTemp(Traversable where,
                                                 Identifier id) {
        return SymbolTools.getTemp(where, id);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getTemp(Identifier id,
                                                 String name) {
        return SymbolTools.getTemp(id, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getTemp
        (Traversable where, Specifier spec, String name) {
        return SymbolTools.getTemp(where, spec, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getTemp(Traversable where,
                                                 List specs, String name) {
        return SymbolTools.getTemp(where, specs, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getArrayTemp
        (Traversable where, List specs, ArraySpecifier aspec, String name) {
        return SymbolTools.getArrayTemp(where, specs, aspec, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getArrayTemp
        (Traversable where, List specs, List aspecs, String name) {
        return SymbolTools.getArrayTemp(where, specs, aspecs, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getPointerTemp(Traversable where,
                                                        Identifier refID) {
        return SymbolTools.getPointerTemp(where, refID);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getPointerTemp
        (Traversable where, List specs, String name) {
        return SymbolTools.getPointerTemp(where, specs, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Identifier getPointerTemp
        (Traversable where, List specs, List pspecs, String name) {
        return SymbolTools.getPointerTemp(where, specs, pspecs, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol> getSymbols(SymbolTable st) {
        return SymbolTools.getSymbols(st);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol>
        getVariableSymbols(SymbolTable st) {
        return SymbolTools.getVariableSymbols(st);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol>
        getGlobalSymbols(Traversable t) {
        return SymbolTools.getGlobalSymbols(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol>
        getParameterSymbols(Procedure proc) {
        return SymbolTools.getParameterSymbols(proc);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol>
        getSideEffectSymbols(FunctionCall fc) {
        return SymbolTools.getSideEffectSymbols(fc);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol>
        getAccessedSymbols(Traversable t) {
        return SymbolTools.getAccessedSymbols(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol> getLocalSymbols(Traversable t) {
        return SymbolTools.getLocalSymbols(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Symbol getSymbolOfName(String name,
                                                     Traversable tr) {
        return SymbolTools.getSymbolOfName(name, tr);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Symbol getSymbolOf(Expression e) {
        return SymbolTools.getSymbolOf(e);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isGlobal(Symbol symbol,
                                               Traversable t) {
        return SymbolTools.isGlobal(symbol, t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isScalar(Symbol symbol) {
        return SymbolTools.isScalar(symbol);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isArray(Symbol symbol) {
        return SymbolTools.isArray(symbol);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isPointer(Symbol symbol) {
        return SymbolTools.isPointer(symbol);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isPointer(Expression e) {
        return SymbolTools.isPointer(e);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isInteger(Symbol symbol) {
        return SymbolTools.isInteger(symbol);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static List getExpressionType(Expression e) {
        return SymbolTools.getExpressionType(e);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static LinkedList getVariableType(Expression e) {
        return SymbolTools.getVariableType(e);
    }

    ////////////////////////////////////////////////////////////////////////////
    // PrintTools
    ////////////////////////////////////////////////////////////////////////////

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printlnStatus(Printable p,
                                                 int min_verbosity) {
        PrintTools.printlnStatus(p, min_verbosity);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printlnStatus(String message,
                                                 int min_verbosity) {
        PrintTools.printlnStatus(message, min_verbosity);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void print(String message, int min_verbosity) {
        PrintTools.print(message, min_verbosity);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void println(String message,
                                           int min_verbosity) {
        PrintTools.println(message, min_verbosity);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printStatus(Printable p,
                                               int min_verbosity) {
        PrintTools.printStatus(p, min_verbosity);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printStatus(String message,
                                               int min_verbosity) {
        PrintTools.printStatus(message, min_verbosity);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void
     printListWithSeparator(List list, PrintWriter w, String sep) {
        PrintTools.printListWithSeparator(list, w, sep);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printListWithComma(List list,
                                                      PrintWriter w) {
        PrintTools.printListWithComma(list, w);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printListWithSpace(List list,
                                                      PrintWriter w) {
        PrintTools.printListWithSpace(list, w);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printList(List list, PrintWriter w) {
        PrintTools.printList(list, w);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void printlnList(List list, PrintWriter w) {
        PrintTools.printlnList(list, w);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static int getVerbosity() {
        return PrintTools.getVerbosity();
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static String collectionToString(Collection coll,
                                                        String separator) {
        return PrintTools.collectionToString(coll, separator);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static String listToString(List list,
                                                  String separator) {
        return PrintTools.listToString(list, separator);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static String listToStringWithSkip(List list,
                                                          String separator) {
        return PrintTools.listToStringWithSkip(list, separator);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static String mapToString(Map map, String separator) {
        return PrintTools.mapToString(map, separator);
    }

    ////////////////////////////////////////////////////////////////////////////
    // DataFlowTools
    ////////////////////////////////////////////////////////////////////////////

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Expression> getUseSet(Traversable t) {
        return DataFlowTools.getUseSet(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void mergeSymbolMaps
        (Map<Symbol, Set<Integer>> orig_map, Map<Symbol,
         Set<Integer>> new_map) {
        DataFlowTools.mergeSymbolMaps(orig_map, new_map);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void mergeMaps
        (Map<Expression, Set<Integer>> orig_map, Map<Expression,
         Set<Integer>> new_map) {
        DataFlowTools.mergeMaps(orig_map, new_map);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Map<Symbol, Set<Integer>>
        convertExprMap2SymbolMap(Map<Expression, Set<Integer>> imap) {
        return DataFlowTools.convertExprMap2SymbolMap(imap);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Map<Symbol,
        Set<Integer>> getUseSymbolMap(Traversable t) {
        return DataFlowTools.getUseSymbolMap(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Map<Symbol,
        Set<Integer>> getDefSymbolMap(Traversable t) {
        return DataFlowTools.getDefSymbolMap(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Section.MAP
        getUseSectionMap(Expression e, RangeDomain rd,
                         Set<Symbol> def_vars) {
        return DataFlowTools.getUseSectionMap(e, rd, def_vars);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Section.MAP
        getDefSectionMap(Expression e, RangeDomain rd,
                         Set<Symbol> def_vars) {
        return DataFlowTools.getDefSectionMap(e, rd, def_vars);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Map<Expression,
        Set<Integer>> getUseMap(Traversable t) {
        return DataFlowTools.getUseMap(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void displayMap(Map<Expression,
                                              Set<Integer>> imap,
                                              String name) {
        DataFlowTools.displayMap(imap, name);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Expression> getDefSet(Traversable t) {
        return DataFlowTools.getDefSet(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static List<Expression> getDefList(Traversable t) {
        return DataFlowTools.getDefList(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Map<Expression,
        Set<Integer>> getDefMap(Traversable t) {
        return DataFlowTools.getDefMap(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Section.MAP getDefSectionMap
        (Traversable t, Map range_map, RangeDomain unioned_rd,
         Set<Symbol> def_vars) {
        return DataFlowTools.getDefSectionMap(t, range_map, unioned_rd,
                                              def_vars);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol> getDefSymbol(Traversable t) {
        return DataFlowTools.getDefSymbol(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Set<Symbol> getUseSymbol(Traversable t) {
        return DataFlowTools.getUseSymbol(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */

    ////////////////////////////////////////////////////////////////////////////
    // IRTools
    ////////////////////////////////////////////////////////////////////////////
    @Deprecated public static boolean containsExpression(Traversable t,
                                                         Expression e) {
        return IRTools.containsExpression(t, e);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static int countExpressions(Traversable t,
                                                   Expression e) {
        return IRTools.countExpressions(t, e);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Expression findExpression(Traversable t,
                                                        Expression e) {
        return IRTools.findExpression(t, e);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean checkConsistency(Traversable t) {
        return IRTools.checkConsistency(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void replaceAll(Traversable t, Expression x,
                                              Expression y) {
        IRTools.replaceAll(t, x, y);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Procedure getParentProcedure(Traversable t) {
        return IRTools.getParentProcedure(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static TranslationUnit
        getParentTranslationUnit(Traversable t) {
        return IRTools.getParentTranslationUnit(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Declaration getLastDeclaration(Traversable t) {
        return IRTools.getLastDeclaration(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static DeclarationStatement
        getLastDeclarationStatement(Traversable t) {
        return IRTools.getLastDeclarationStatement(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Statement
        getFirstNonDeclarationStatement(Traversable t) {
        return IRTools.getFirstNonDeclarationStatement(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static Expression
        replaceSymbol(Expression e, Symbol var, Expression expr) {
        return IRTools.replaceSymbol(e, var, expr);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void replaceSymbolIn(Traversable t,
                                                   Symbol var,
                                                   Expression e) {
        IRTools.replaceSymbolIn(t, var, e);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean
        containsExpressions(Traversable t,
                            Collection<? extends Expression> es) {
        return IRTools.containsExpressions(t, es);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsSymbol(Traversable t,
                                                     Symbol var) {
        return IRTools.containsSymbol(t, var);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsSymbols(Traversable t,
                                                      Set<Symbol> vars) {
        return IRTools.containsSymbols(t, vars);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsSymbols(Set<Symbol> vars,
                                                      Set<Symbol> symbols) {
        return IRTools.containsSymbols(vars, symbols);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsClass(Traversable t,
                                    Class <? extends Traversable> type) {
        return IRTools.containsClass(t, type);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsClasses(Traversable t,
                                    Set<Class<? extends Traversable>> types) {
        return IRTools.containsClasses(t, types);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsBinary(Traversable t,
                                                     BinaryOperator op) {
        return IRTools.containsBinary(t, op);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsUnary(Traversable t,
                                                    UnaryOperator op) {
        return IRTools.containsUnary(t, op);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static List<UnaryExpression>
        getUnaryExpression(Traversable t, UnaryOperator op) {
        return IRTools.getUnaryExpression(t, op);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsSideEffect(Traversable t) {
        return IRTools.containsSideEffect(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static <T extends Traversable>
        T getAncestorOfType(Traversable t, Class<T> type) {
        return IRTools.getAncestorOfType(t, type);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static <T extends Traversable> List<T>
        getDescendentsOfType(Traversable t, Class<T> type) {
        return IRTools.getDescendentsOfType(t, type);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isAncestorOf(Traversable anc,
                                                   Traversable des) {
        return IRTools.isAncestorOf(anc, des);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean isDescendantOf(Traversable des,
                                                     Traversable anc) {
        return IRTools.isDescendantOf(des, anc);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static boolean containsFunctionCall(Traversable t) {
        return IRTools.containsFunctionCall(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static List<FunctionCall>
        getFunctionCalls(Traversable t) {
        return IRTools.getFunctionCalls(t);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static <T extends PragmaAnnotation> List<T>
        collectPragmas(Traversable t, Class<T> pragma_cls, String key) {
        return IRTools.collectPragmas(t, pragma_cls, key);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static List<Statement>
        getLastStatements(Procedure proc) {
        return IRTools.getLastStatements(proc);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static <T extends Statement> List<T>
        getStatementsOfType(Traversable t, Class<T> type) {
        return IRTools.getStatementsOfType(t, type);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static <T extends Expression> List<T>
        getExpressionOfType(Traversable t, Class<T> type) {
        return IRTools.getExpressionsOfType(t, type);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static List<Procedure>
        getProcedureList(Program program) {
        return IRTools.getProcedureList(program);
    }

    /**
    * @deprecated As of release 1.2; use the new tools in
    * {@link DataFlowTools}, {@link IRTools}, {@link PrintTools}, or
    * {@link SymbolTools}.
    */
    @Deprecated public static void
        removeAnnotations(Traversable t, Class<? extends Annotation> type) {
        IRTools.removeAnnotations(t, type);
    }

    ////////////////////////////////////////////////////////////////////////////
    // General Tools
    ////////////////////////////////////////////////////////////////////////////

    public static int indexByReference(List list, Object obj) {
        int index = 0;
        Iterator iter = list.iterator();
        while (iter.hasNext()) {
            if (iter.next() == obj) {
                return index;
            } else {
                index++;
            }
        }
        return -1;
    }

    public static int identityIndexOf(List l, Object o) {
        int size = l.size();
        for (int i = 0; i < size; i++) {
            if (l.get(i) == o) {
                return i;
            }
        }
        return -1;
    }

    public static boolean verifyHomogeneousList(List list, Class type) {
        Iterator iter = list.iterator();
        while (iter.hasNext()) {
            if (!type.isInstance(iter.next())) {
                return false;
            }
        }
        return true;
    }

    public static boolean containsClass(Collection c, Class type) {
        if (c == null) {
            return false;
        }
        for (Object o : c) {
            if (o.getClass() == type) {
                return true;
            }
        }
        return false;
    }

    /**
    * Returns a current system time in seconds since a system-wise reference
    * time.
    *
    * @return the time in seconds.
    */
    public static double getTime() {
        return (System.currentTimeMillis() / 1000.0);
    }

    /**
    * Returns the elapsed time in seconds since the given reference time.
    *
    * @param since the reference time
    * @return the elapsed time in seconds
    */
    public static double getTime(double since) {
        return (System.currentTimeMillis() / 1000.0 - since);
    }

    /**
    * Invokes a exit handler after printing the specified message.
    * @param msg the message to be printed
    */
    public static void exit(String msg) {
        System.out.println(msg);
        Tools.exit(0);
    }

    /** Flag for selecting how exit() is handled. */
    private static boolean exit_throws_exception = false;

    /**
    * Selects how Tools.exit() behaves.
    * Setting <b>flag = true</b> causes <b>Tools.exit()</b> throw a runtime
    * exception instead of invoking <b>System.exit()</b>.
    * @param flag the boolean flag
    */
    public static void exitThrowsException(boolean flag) {
        exit_throws_exception = flag;
    }

    /**
    * Invokes exit operation.
    * Depending on the flag set by {@link #exitThrowsException(boolean)}, this
    * method either calls {@link System#exit(int)} or throws a
    * {@link RuntimeException}.
    * @param status the exit status
    * @exception RuntimeException if the last call to
    *       {@link #exitThrowsException(boolean)} is with <b>true</b>.
    */
    public static void exit(int status) {
        if (exit_throws_exception) {
            throw new RuntimeException("Exiting with status " + status);
        } else {
            System.exit(status);
        }
    }

    /**
    * Wrapper for Collection.addAll() that suppress "unchecked" warnings.
    * Pass writers are responsible for assuring type safety when using this
    * wrapper.
    */
    @SuppressWarnings("unchecked")
    public static <T> boolean addAll(Collection<T> c1, Collection c2) {
        return c1.addAll(c2);
    }

    /**
    * Wrapper for Collection.containsAll() that suppress "unchecked" warnings.
    * Pass writers are responsible for assuring type safety when using this
    * wrapper.
    */
    @SuppressWarnings("unchecked")
    public static <T> boolean containsAll(Collection<T> c1, Collection c2) {
        return c1.containsAll(c2);
    }

    /**
    * Wrapper for Collection.removeAll() that suppress "unchecked" warnings.
    * Pass writers are responsible for assuring type safety when using this
    * wrapper.
    */
    @SuppressWarnings("unchecked")
    public static <T> boolean removeAll(Collection<T> c1, Collection c2) {
        return c1.removeAll(c2);
    }

    /**
    * Wrapper for Collection.retainAll() that suppress "unchecked" warnings.
    * Pass writers are responsible for assuring type safety when using this
    * wrapper.
    */
    @SuppressWarnings("unchecked")
    public static <T> boolean retainAll(Collection<T> c1, Collection c2) {
        return c1.retainAll(c2);
    }

}

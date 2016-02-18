package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.analysis.Domain;
import cetus.analysis.IPPointsToAnalysis;
import cetus.analysis.PointsToDomain;
import cetus.analysis.PointsToRel;
import cetus.hir.*;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Provides tools for Def-Use or Use-Def chain computation.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class ChainTools {

    public static boolean isDefinedArrayDeclarator(VariableDeclarator vd) {
        if (vd.getArraySpecifiers().size() > 0 && vd.getInitializer() != null) {
//            System.out.println("isDefinedArrayDeclarator is true: " + vd.toString());
            return true;
        } else {
            return false;
        }
    }

    public static boolean isDefinedArrayNestedDeclarator(NestedDeclarator vd) {
        if (vd.getArraySpecifiers().size() > 0 && vd.getInitializer() != null) {
//            System.out.println("isDefinedArrayDeclarator is true: " + vd.toString());
            return true;
        } else {
            return false;
        }
    }

    public static String[] expandArguments(String[] args, String suffix) {
        if (args == null) {
            return args;
        }
        if (args.length == 0) {
            return args;
        }
        ArrayList<String> argsList = new ArrayList<String>();
        final String extension = "." + suffix;
        FilenameFilter filter = new FilenameFilter() {

            public boolean accept(File dir, String name) {
                if (name.endsWith(extension)) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        Set<File> dirSet = new HashSet<File>();
        for (String s : args) {
            File file = new File(s);
            if (file.exists()) {
                // process file
                if (file.isDirectory()) {
                    // expand
                    List<File> fileList = new ArrayList<File>();
                    listDirectory(fileList, dirSet, s, filter);
                    for (File f : fileList) {
                        try {
                            argsList.add(f.getCanonicalPath());
                        } catch (IOException ex) {
                            Logger.getLogger(ChainTools.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }
                } else {
                    argsList.add(s);
                }
            } else {
                argsList.add(s);
            }
        }
        return (String[]) argsList.toArray(args);
    }

    public static void listDirectory(List<File> fileList, Set<File> dirSet, String directory, FilenameFilter filter) {
        // handle fiels
        File dir = new File(directory);
        dirSet.add(dir);
        if (!dir.isDirectory()) {
            throw new IllegalArgumentException("FileUtils: no such directory: " + directory);
        }
        File[] files = dir.listFiles(filter);
        Arrays.sort(files);
        for (File file : files) {
            fileList.add(file);
        }
        // handle directories
        files = dir.listFiles(new FilenameFilter() {

            public boolean accept(File dir, String name) {
                File f = new File(dir + "/" + name);
                if (f.isDirectory()) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        for (File file : files) {
            try {
                listDirectory(fileList, dirSet, file.getCanonicalPath(), filter);
            } catch (IOException ex) {
                Logger.getLogger(ChainTools.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    public static boolean isNonConstReferenceTypeParameter(VariableDeclaration varDec) {
        List<Specifier> specList = varDec.getSpecifiers();
        for (Specifier sp : specList) {
            if (sp.toString().equals("const")) {
                return false;
            }
        }
        Declarator d = varDec.getDeclarator(0);
        if (d instanceof VariableDeclarator) {
            VariableDeclarator vd = null;
            if (d instanceof VariableDeclarator) {
                vd = (VariableDeclarator) d;
            } else {
                return false;
            }
            // Collect the array and pointer type of parameters
            if (vd.getArraySpecifiers().size() > 0) {
                return true;
            } else {
                List type = vd.getTypeSpecifiers();
                for (Object oo : type) {
                    if (oo instanceof PointerSpecifier) { // pointer type
                        return true;
                    }
                }
            }
            return false;
        } else if (d instanceof NestedDeclarator) {
            NestedDeclarator vd = null;
            if (d instanceof NestedDeclarator) {
                vd = (NestedDeclarator) d;
            } else {
                return false;
            }
            // Collect the array and pointer type of parameters
            if (vd.getArraySpecifiers().size() > 0) {
                return true;
            } else {
                List type = vd.getTypeSpecifiers();
                for (Object oo : type) {
                    if (oo instanceof PointerSpecifier) { // pointer type
                        return true;
                    }
                }
            }
            return false;
        } else {
            return false;
        }
    }
    
    public static List<Expression> getRefIDExpressionListInConditionalExpressionArg(ConditionalExpression conEx, Procedure proc) {
    	List<Expression> retList = new ArrayList<Expression>();
    	// true expression
    	Expression trueEx = conEx.getTrueExpression();
    	if (trueEx instanceof ConditionalExpression) {
    		retList.addAll(getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)trueEx, proc));
    	} else {
        	IDExpression trueID = getRefIDExpression(trueEx, proc);
        	if (trueID != null) {
        		retList.add(trueID);
        	}
    	}
    	// false expression
    	Expression falseEx = conEx.getFalseExpression();
    	if (falseEx instanceof ConditionalExpression) {
    		retList.addAll(getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)falseEx, proc));
    	} else {
	    	IDExpression falseID = getRefIDExpression(falseEx, proc);
	    	if (falseID != null) {
	    		retList.add(falseID);
	    	}
    	}
    	return retList;
    }

    public static void traverseIR(Traversable tr) {
        System.out.println("######## Traversing " + tr.toString() + " ##############");
        DepthFirstIterator iter = new DepthFirstIterator(tr);
        while (iter.hasNext()) {
            Object o = iter.next();
            System.out.println("o: " + o.toString() + ", class: " + o.getClass().getCanonicalName());
        }
    }

    public static Procedure getParentProcedure(Traversable trav, Program program) {
        Traversable t = trav;
        while (true) {
            if (t instanceof Procedure) {
                break;
            }
            t = t.getParent();
            if (t == null) {
                break;
            }
        }
        if (t != null) {
            return (Procedure) t;
        }
        IDExpression idEx = null;
        if (trav instanceof Expression) {
            idEx = getIDExpression((Expression) trav);
        } else if (trav instanceof VariableDeclarator) {
            idEx = getIDExpression(((VariableDeclarator) trav).getID());
        } else if (trav instanceof NestedDeclarator) {
            idEx = getIDExpression(((NestedDeclarator) trav).getID());
        }
        if (idEx != null) {
            // check if t is the formal parameter and return the procedure with the parameter
            Set<Procedure> procList = getProcedureSet(program);
            for (Procedure proc : procList) {
                List<VariableDeclaration> paramList = proc.getParameters();
                for (VariableDeclaration vd : paramList) {
                    if (vd.getDeclarator(0) instanceof NestedDeclarator) {
                        NestedDeclarator declarator = (NestedDeclarator) vd.getDeclarator(0);
                        if (System.identityHashCode(idEx) == System.identityHashCode(declarator.getID())) {
                            return proc;
                        }
                    } else {
                        VariableDeclarator declarator = (VariableDeclarator) vd.getDeclarator(0);
                        if (System.identityHashCode(idEx) == System.identityHashCode(declarator.getID())) {
                            return proc;
                        }
                    }
                }
            }
        }
        return (Procedure) t;
    }

    public static boolean hasSameToString(Expression ex1, Expression ex2) {
        if (ex1 == null || ex2 == null) {
            return false;
        }
        return ex1.toString().equals(ex2.toString());
    }

    public static boolean hasSameStructureVariableIdentifier(Expression def, Expression use, Procedure proc) {
        Expression varEx1 = getIDVariableOnlyInStruct(def);
        Expression varEx2 = getIDVariableOnlyInStruct(use);
        return hasSameToString(varEx1, varEx2);
    }

    public static boolean hasSameArrayAccessInStruct(Expression def, Expression use, Procedure proc) {
        Expression defEx = getMemberOnlyInStruct(def);
        Expression useEx = null;
        if (defEx == null) {
            defEx = getIDVariableOnlyInStruct(def);
            useEx = getIDVariableOnlyInStruct(use);
            return hasSameToString(defEx, useEx);
        } else {
            defEx = getIDVariablePlusMemberInStruct(def);
            if (defEx == null) {
                return false;
            } else {
                useEx = getIDVariablePlusMemberInStruct(use);
                return hasSameToString(defEx, useEx);
            }
        }
    }

    public static boolean hasSameArrayIdentifier(Expression ex1, Expression ex2) {
        if (!isArrayAccess(ex1) || !isArrayAccess(ex2)) {
            return false;
        }
        Expression id1 = getIDExpression(ex1);
        Expression id2 = getIDExpression(ex2);
        return hasSameToString(id1, id2);
    }

    public static boolean isArrayAccessWithConstantIndex(Expression expression) {
        if (isArrayAccess(expression) == false || isArrayAccessWithNoIndex(expression)) {
            return false;
        }
        DepthFirstIterator irIter = new DepthFirstIterator(expression);
        irIter.pruneOn(ArrayAccess.class);
        Object o = irIter.next(); // ArrayAccess
        if (o instanceof ArrayAccess) {
            int arrayDim = 0;
            Expression arrayName = ((ArrayAccess) o).getArrayName();
            if (arrayName instanceof Identifier) {
                if (SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size() != 1) {
                    throw new RuntimeException("SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size(): " + SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size());
                }
                ArraySpecifier arraySpec = (ArraySpecifier) SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().get(0);
                arrayDim = arraySpec.getNumDimensions();
            } else if (arrayName instanceof AccessExpression) {
                Expression rhsEx = ((AccessExpression) arrayName).getRHS();
                if (SymbolTools.getSymbolOf(rhsEx).getArraySpecifiers().size() != 1) {
                    throw new RuntimeException("SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size(): " + SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size());
                }
                ArraySpecifier arraySpec = (ArraySpecifier) SymbolTools.getSymbolOf(rhsEx).getArraySpecifiers().get(0);
                arrayDim = arraySpec.getNumDimensions();
            } else if (arrayName instanceof UnaryExpression) {
                arrayName = ((UnaryExpression) arrayName).getExpression();
                if (SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size() != 1) {
                    throw new RuntimeException("SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size(): " + SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size());
                }
                ArraySpecifier arraySpec = (ArraySpecifier) SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().get(0);
                arrayDim = arraySpec.getNumDimensions();
            } else {
                throw new RuntimeException("Unexpected type for ArrayName: " + arrayName + ", class: " + arrayName.getClass().getCanonicalName() + ", proc: " + IRTools.getParentProcedure(arrayName));
            }
            int actualDim = 0;
            List<Expression> indexList = ((ArrayAccess) o).getIndices();
            boolean hasOnlyLiteral = true;
            for (Expression indexEx : indexList) {
                if (indexEx instanceof Identifier) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof BinaryExpression) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof IntegerLiteral) {
                    actualDim++;
                } else if (indexEx instanceof ArrayAccess) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof UnaryExpression) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof Typecast) {
                    if (containsIDEx(((Typecast) indexEx).getExpression())) {
                        hasOnlyLiteral = false;
                    }
                } else if (indexEx instanceof ConditionalExpression) {
                    // Temp : may need to be modified
                    hasOnlyLiteral = false;
                    // dyn_dtree[((dist<256) ? dist_code[dist] : dist_code[(256+(dist>>7))])]
                } else if (indexEx instanceof FunctionCall) {
                    // global_regs[subreg_regno(x)] (gcc)
                    hasOnlyLiteral = false;
                } else {
                    throw new RuntimeException("not ID: " + indexEx.toString() + ", class: " + indexEx.getClass().getCanonicalName() + ", expr: " + expression);
                }
            }
            if (hasOnlyLiteral && arrayDim == actualDim) {
                return true;
            } else {
//                if (hasOnlyLiteral && arrayDim != actualDim) {
//                    throw new RuntimeException("hasOnlyLiteral && arrayDim != actualDim: " + expression.toString() + ", arrayDim: " + arrayDim + ", actualDim: " + actualDim + ", proc: " + IRTools.getParentProcedure(expression));
//                }
                return false;
            }
        } else {
            ChainTools.traverseIR((Traversable) o);
            throw new RuntimeException("not ArrayAccess: " + o.toString() + ", class: " + o.getClass().getCanonicalName() + ", expr: " + expression);
        }
    }

    public static boolean isArrayAccessWithPartiallyConstantIndex(Expression expression) {
        if (isArrayAccess(expression) == false || isArrayAccessWithNoIndex(expression)) {
            return false;
        }
        DepthFirstIterator irIter = new DepthFirstIterator(expression);
        irIter.pruneOn(ArrayAccess.class);
        Object o = irIter.next(); // ArrayAccess
        if (o instanceof ArrayAccess) {
            int arrayDim = 0;
            Expression arrayName = ((ArrayAccess) o).getArrayName();
            if (arrayName instanceof Identifier) {
                if (SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size() != 1) {
                    throw new RuntimeException("SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size(): " + SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size());
                }
                ArraySpecifier arraySpec = (ArraySpecifier) SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().get(0);
                arrayDim = arraySpec.getNumDimensions();
            } else if (arrayName instanceof AccessExpression) {
                Expression rhsEx = ((AccessExpression) arrayName).getRHS();
                if (SymbolTools.getSymbolOf(rhsEx).getArraySpecifiers().size() != 1) {
                    throw new RuntimeException("SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size(): " + SymbolTools.getSymbolOf(arrayName).getArraySpecifiers().size());
                }
                ArraySpecifier arraySpec = (ArraySpecifier) SymbolTools.getSymbolOf(rhsEx).getArraySpecifiers().get(0);
                arrayDim = arraySpec.getNumDimensions();
            } else {
                throw new RuntimeException("Unexpected type for ArrayName: " + arrayName + ", class: " + arrayName.getClass().getCanonicalName());
            }
            int actualDim = 0;
            List<Expression> indexList = ((ArrayAccess) o).getIndices();
            boolean hasOnlyLiteral = true;
            for (Expression indexEx : indexList) {
                if (indexEx instanceof Identifier) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof BinaryExpression) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof IntegerLiteral) {
                    actualDim++;
                } else if (indexEx instanceof ArrayAccess) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof UnaryExpression) {
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof Typecast) {
                    if (containsIDEx(((Typecast) indexEx).getExpression())) {
                        hasOnlyLiteral = false;
                    }
                } else if (indexEx instanceof ConditionalExpression) {
                    // Temp : may need to be modified
                    hasOnlyLiteral = false;
                } else if (indexEx instanceof FunctionCall) {
                    // global_regs[subreg_regno(x)] (gcc)
                    hasOnlyLiteral = false;
                } else {
                    throw new RuntimeException("not ID: " + indexEx.toString() + ", class: " + indexEx.getClass().getCanonicalName() + ", expr: " + expression);
                }
            }
            if (hasOnlyLiteral && arrayDim != actualDim) {
                return true;
            } else {
                return false;
            }
        } else {
            ChainTools.traverseIR((Traversable) o);
            throw new RuntimeException("not ArrayAccess: " + o.toString() + ", class: " + o.getClass().getCanonicalName() + ", expr: " + expression);
        }
    }

    public static boolean isArrayAccessWithVariableIndex(Expression expression) {
        if (isArrayAccess(expression) == false || isArrayAccessWithNoIndex(expression)) {
            return false;
        }
        DepthFirstIterator irIter = new DepthFirstIterator(expression);
        Object o = irIter.next(); // ArrayAccess
        if (o instanceof ArrayAccess) {
            List<Expression> indexList = ((ArrayAccess) o).getIndices();
            boolean hasVariable = false;
            for (Expression indexEx : indexList) {
                if (indexEx instanceof Identifier) {
                    hasVariable = true;
                } else if (indexEx instanceof BinaryExpression) {
                    hasVariable = true;
                } else if (indexEx instanceof IntegerLiteral) {
                    // no change for hasVariable
                } else if (indexEx instanceof ArrayAccess) {
                    hasVariable = true;
                } else if (indexEx instanceof UnaryExpression) {
                    hasVariable = containsIDEx(((UnaryExpression) indexEx).getExpression());
                } else if (indexEx instanceof Typecast) {
                    hasVariable = containsIDEx(((Typecast) indexEx).getExpression());
                } else if (indexEx instanceof ConditionalExpression) {
                    // Temp : may need to be modified
                    hasVariable = true;
                } else if (indexEx instanceof ConditionalExpression) {
                    // global_regs[subreg_regno(x)] (gcc)
                    hasVariable = true;
                } else {
                    throw new RuntimeException("not ID: " + indexEx.toString() + ", class: " + indexEx.getClass().getCanonicalName() + ", expr: " + expression);
                }
            }
            return hasVariable;
        } else {
            ChainTools.traverseIR((Traversable) o);
            throw new RuntimeException("not ArrayAccess: " + o.toString() + ", class: " + o.getClass().getCanonicalName() + ", expr: " + expression);
        }
    }

    public static boolean isArrayAccessWithNoIndex(Expression expression) {
        return isArrayAccess(expression) && !(expression instanceof ArrayAccess);
    }

    public static boolean isArrayAccess(Expression expression) {
        if (expression == null) {
            return false;
        }
        Symbol sym = SymbolTools.getSymbolOf(expression);
        if (sym != null) {
            return SymbolTools.isArray(sym);
        } else {
            Traversable parent = expression.getParent();
            if (parent instanceof VariableDeclarator) {
                VariableDeclarator vd = (VariableDeclarator) expression.getParent();
                List arraySpecList = vd.getArraySpecifiers();
                if (arraySpecList != null) {
                    return true;
                } else {
                    return false;
                }
            } else if (parent instanceof NestedDeclarator) {
                NestedDeclarator vd = (NestedDeclarator) expression.getParent();
                List arraySpecList = vd.getArraySpecifiers();
                if (arraySpecList != null) {
                    return true;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    public static boolean isArrayAccessInStruct(Expression expression, Procedure proc) {
        if (isStructureAccess(expression, proc) == false) {
            return false;
        }
        if (expression instanceof ArrayAccess) {
            return true;
        } else {
            return false;
        }
    }

    public static List<Expression> getDefListInDec(VariableDeclarator declarator) {
        final int LHS_NAMEID_IDX = 1, RHS_IDX = 3;
        List<Expression> defList = new ArrayList<Expression>();
        if (declarator == null) {
            return defList;
        }
        boolean arrayDec = isDefinedArrayDeclarator(declarator);
        DepthFirstIterator vdIter = new DepthFirstIterator(declarator);
        int idx = 0;
        while (vdIter.hasNext()) {
            Object vdo = vdIter.next();
            if (idx == LHS_NAMEID_IDX) {
                if (vdo instanceof NameID) {
                    // var = XXX ==> LHS is NameID
                    if (arrayDec) {
                        defList.add((Expression) vdo);
                    } else {
                        if (vdIter.hasNext() && vdIter.next() instanceof Initializer) {
                            defList.add((Expression) vdo);
                            vdIter.reset();
                            vdIter.next();  // whole
                            vdIter.next();  // NameID
                        }
                    }
                }
            } else {
                if (arrayDec) {
                    if (vdo instanceof AssignmentExpression) {
                        defList.add(((AssignmentExpression) vdo).getLHS());
                    } else if (vdo instanceof UnaryExpression) {
                        UnaryExpression ue = (UnaryExpression) vdo;
                        UnaryOperator uop = ue.getOperator();
                        if (uop == UnaryOperator.POST_DECREMENT ||
                                uop == UnaryOperator.POST_INCREMENT ||
                                uop == UnaryOperator.PRE_DECREMENT ||
                                uop == UnaryOperator.PRE_INCREMENT) {
                            defList.add(ue.getExpression());
                        }
                    } else if (vdo instanceof FunctionCall) {
                        // TODO
                        // handle the case that
                        // declarator contains a function call which
                        // modifies the parameters
                    }
                } else {
                    if (idx == RHS_IDX && vdo instanceof AssignmentExpression) {
                        defList.add(((AssignmentExpression) vdo).getLHS());
                    } else if (idx == RHS_IDX && vdo instanceof UnaryExpression) {
                        UnaryExpression ue = (UnaryExpression) vdo;
                        UnaryOperator uop = ue.getOperator();
                        if (uop == UnaryOperator.POST_DECREMENT ||
                                uop == UnaryOperator.POST_INCREMENT ||
                                uop == UnaryOperator.PRE_DECREMENT ||
                                uop == UnaryOperator.PRE_INCREMENT) {
                            defList.add(ue.getExpression());
                        }
                    } else if (idx == RHS_IDX && vdo instanceof FunctionCall) {
                        // TODO
                        // handle the case that
                        // declarator contains a function call which
                        // modifies the parameters
                    }
                }
            }
            idx++;
        }
        return defList;
    }

    public static List<Expression> getDefListInNestedDec(NestedDeclarator declarator) {
        final int LHS_NAMEID_IDX = 1, RHS_IDX = 3;
        List<Expression> defList = new ArrayList<Expression>();
        if (declarator == null) {
            return defList;
        }
        boolean arrayDec = isDefinedArrayNestedDeclarator(declarator);
        DepthFirstIterator vdIter = new DepthFirstIterator(declarator);
        int idx = 0;
        while (vdIter.hasNext()) {
            Object vdo = vdIter.next();
            if (idx == LHS_NAMEID_IDX) {
                if (vdo instanceof NameID) {
                    // var = XXX ==> LHS is NameID
                    if (arrayDec) {
                        defList.add((Expression) vdo);
                    } else {
                        if (vdIter.hasNext() && vdIter.next() instanceof Initializer) {
                            defList.add((Expression) vdo);
                            vdIter.reset();
                            vdIter.next();  // whole
                            vdIter.next();  // NameID
                        }
                    }
                }
            } else {
                if (arrayDec) {
                    if (vdo instanceof AssignmentExpression) {
                        defList.add(((AssignmentExpression) vdo).getLHS());
                    } else if (vdo instanceof UnaryExpression) {
                        UnaryExpression ue = (UnaryExpression) vdo;
                        UnaryOperator uop = ue.getOperator();
                        if (uop == UnaryOperator.POST_DECREMENT ||
                                uop == UnaryOperator.POST_INCREMENT ||
                                uop == UnaryOperator.PRE_DECREMENT ||
                                uop == UnaryOperator.PRE_INCREMENT) {
                            defList.add(ue.getExpression());
                        }
                    } else if (vdo instanceof FunctionCall) {
                        // TODO
                        // handle the case that
                        // declarator contains a function call which
                        // modifies the parameters
                    }
                } else {
                    if (idx == RHS_IDX && vdo instanceof AssignmentExpression) {
                        defList.add(((AssignmentExpression) vdo).getLHS());
                    } else if (idx == RHS_IDX && vdo instanceof UnaryExpression) {
                        UnaryExpression ue = (UnaryExpression) vdo;
                        UnaryOperator uop = ue.getOperator();
                        if (uop == UnaryOperator.POST_DECREMENT ||
                                uop == UnaryOperator.POST_INCREMENT ||
                                uop == UnaryOperator.PRE_DECREMENT ||
                                uop == UnaryOperator.PRE_INCREMENT) {
                            defList.add(ue.getExpression());
                        }
                    } else if (idx == RHS_IDX && vdo instanceof FunctionCall) {
                        // TODO
                        // handle the case that
                        // declarator contains a function call which
                        // modifies the parameters
                    }
                }
            }
            idx++;
        }
        return defList;
    }

    public static List<Expression> getUseListInDec(VariableDeclarator declarator) {
        // This is copied from DataFlowTools.getUseSet()
        List<Expression> ret = new ArrayList<Expression>();

        DepthFirstIterator iter = new DepthFirstIterator(declarator);

        // Handle these expressions specially.
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(AssignmentExpression.class);

        while (iter.hasNext()) {
            Object o = iter.next();

            if (o instanceof AccessExpression) {
                AccessExpression ae = (AccessExpression) o;
                DepthFirstIterator ae_iter = new DepthFirstIterator(ae);
                iter.pruneOn(ArrayAccess.class);

                // Catches array subscripts in the access expression.
                while (ae_iter.hasNext()) {
                    Object oo = ae_iter.next();
                    if (oo instanceof ArrayAccess) {
                        ArrayAccess aa = (ArrayAccess) oo;
                        Set aa_use = DataFlowTools.getUseSet(aa);
                        aa_use.remove(aa);
                        ret.addAll(aa_use);
                    }
                }

                ret.add(ae);
            } else if (o instanceof ArrayAccess) {
                ArrayAccess aa = (ArrayAccess) o;

                for (int i = 0; i < aa.getNumIndices(); ++i) {
                    ret.addAll(DataFlowTools.getUseSet(aa.getIndex(i)));
                }

                ret.add(aa);
            } else if (o instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression) o;
                ret.addAll(DataFlowTools.getUseSet(ae.getRHS()));
                Set lhs_use = DataFlowTools.getUseSet(ae.getLHS());

                // Other cases should include the lhs in the used set. (+=,...)
                if (ae.getOperator() == AssignmentOperator.NORMAL) {
                    lhs_use.remove(ae.getLHS());
                }

                ret.addAll(lhs_use);
            } else if (o instanceof Identifier) {
                Identifier id = (Identifier) o;

                if (id.getSymbol() instanceof Procedure ||
                        id.getSymbol() instanceof ProcedureDeclarator); else {
                    ret.add(id);
                }
            }
        }

        return ret;
    }

    public static List<Expression> getUseListInNestedDec(NestedDeclarator declarator) {
        // This is copied from DataFlowTools.getUseSet()
        List<Expression> ret = new ArrayList<Expression>();

        DepthFirstIterator iter = new DepthFirstIterator(declarator);

        // Handle these expressions specially.
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(AssignmentExpression.class);

        while (iter.hasNext()) {
            Object o = iter.next();

            if (o instanceof AccessExpression) {
                AccessExpression ae = (AccessExpression) o;
                DepthFirstIterator ae_iter = new DepthFirstIterator(ae);
                iter.pruneOn(ArrayAccess.class);

                // Catches array subscripts in the access expression.
                while (ae_iter.hasNext()) {
                    Object oo = ae_iter.next();
                    if (oo instanceof ArrayAccess) {
                        ArrayAccess aa = (ArrayAccess) oo;
                        Set aa_use = DataFlowTools.getUseSet(aa);
                        aa_use.remove(aa);
                        ret.addAll(aa_use);
                    }
                }

                ret.add(ae);
            } else if (o instanceof ArrayAccess) {
                ArrayAccess aa = (ArrayAccess) o;

                for (int i = 0; i < aa.getNumIndices(); ++i) {
                    ret.addAll(DataFlowTools.getUseSet(aa.getIndex(i)));
                }

                ret.add(aa);
            } else if (o instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression) o;
                ret.addAll(DataFlowTools.getUseSet(ae.getRHS()));
                Set lhs_use = DataFlowTools.getUseSet(ae.getLHS());

                // Other cases should include the lhs in the used set. (+=,...)
                if (ae.getOperator() == AssignmentOperator.NORMAL) {
                    lhs_use.remove(ae.getLHS());
                }

                ret.addAll(lhs_use);
            } else if (o instanceof Identifier) {
                Identifier id = (Identifier) o;

                if (id.getSymbol() instanceof Procedure ||
                        id.getSymbol() instanceof ProcedureDeclarator); else {
                    ret.add(id);
                }
            }
        }

        return ret;
    }

    public static void setGenBit(DFANode cfgNode, AnalysisTarget defMapEntry[], Expression currentEx) {
        for (int i = 0; i < defMapEntry.length; i++) {
            if (currentEx.toString().contains("argv") &&
                    defMapEntry[i].getExpression().toString().contains("argv")) {
//                System.out.println("setGenBit: def: " + defMapEntry[i].getExpression() + ", and currentEx: " + currentEx + ", matchForGen: " + ChainTools.matchExpressionForGen(defMapEntry[i].getExpression(), currentEx));
            }
            if (ChainTools.matchExpressionForGen(defMapEntry[i].getExpression(), currentEx)) {
                // compare Hash Code between current Expression and Def Mapping
                BitSet genBitSet = cfgNode.getData("GenSet");
                genBitSet.set(i);
                cfgNode.putData("GenSet", genBitSet);
            }
        }
    }

    public static boolean isStructureAccessInParam(Expression currentEx, Procedure proc) {
        Set<Symbol> paramSym = SymbolTools.getParameterSymbols(proc);
        for (Symbol s : paramSym) {
            if (s.getSymbolName().equals(currentEx.toString())) {
                return SymbolTools.isStruct(s, proc);
            }
        }
        return false;
    }

    public static boolean isStructureAccessInDec(Expression currentEx, Procedure proc) {
        Set<Symbol> symSet = SymbolTools.getLocalSymbols(proc);
        for (Symbol s : symSet) {
            if (s.getSymbolName().equals(currentEx.toString())) {
                return SymbolTools.isStruct(s, proc);
            }
        }
        return false;
    }

    public static boolean isStructureAccess(Expression currentEx, Procedure proc) {
        if (currentEx == null) {
            return false;
        } else if (SymbolTools.getSymbolOf(currentEx) == null) {
            if (isStructureAccessInParam(currentEx, proc)) {
                return true;
            } else {
                if (isStructureAccessInDec(currentEx, proc)) {
                    return true;
                } else {
                    return false;
                }
            }
        }
        return SymbolTools.isStruct(SymbolTools.getSymbolOf(currentEx), proc);
    }

    public static boolean isPointerAccess(Expression currentEx) {
        Expression id = getIDExpression(currentEx);
        if (id != null && id instanceof Identifier) {
            return SymbolTools.isPointer(((Identifier) id).getSymbol());
        }
        return SymbolTools.isPointer(currentEx);
    }

    public static Expression getIDVariablePlusMemberInStruct(Expression currentEx) {
        if (currentEx instanceof Identifier) {
            return currentEx;
        } else if (currentEx instanceof NameID) {
            return currentEx;
        } else if (currentEx instanceof ArrayAccess) {
            if (isArrayAccessWithConstantIndex(currentEx)) {
                return currentEx;
            } else {
                return getIDVariablePlusMemberInStruct(((ArrayAccess) currentEx).getArrayName());
            }
        } else if (currentEx instanceof AccessExpression) {
            return currentEx;
        } else if (currentEx instanceof UnaryExpression) {
            return getIDVariablePlusMemberInStruct(((UnaryExpression) currentEx).getExpression());
        } else if (currentEx instanceof Typecast) {
            return getIDVariablePlusMemberInStruct(((Typecast) currentEx).getExpression());
        } else {
            throw new RuntimeException("Unhandled Type : " + currentEx + ", class: " + currentEx.getClass().getCanonicalName());
        }
    }

    public static Expression getIDVariableOnlyInStruct(Expression currentEx) {
        if (currentEx instanceof Identifier) {
            return (Identifier) currentEx;
        } else if (currentEx instanceof AccessExpression) {
            return getIDVariableOnlyInStruct(((AccessExpression) currentEx).getLHS());
        } else if (currentEx instanceof UnaryExpression) {
            return getIDVariableOnlyInStruct(((UnaryExpression) currentEx).getExpression());
        } else if (currentEx instanceof ArrayAccess) {
            return getIDVariableOnlyInStruct(((ArrayAccess) currentEx).getArrayName());
        } else if (currentEx instanceof NameID) {
            return (NameID) currentEx;
        } else if (currentEx instanceof Typecast) {
            return getIDVariableOnlyInStruct(((Typecast) currentEx).getExpression());
        } else if (currentEx instanceof BinaryExpression) {
            List<Expression> useList = getUseList(currentEx);
            if (useList.size() > 1) {
                for (Expression expInUseList : useList) {
//                    System.out.println("expInUseList: " + expInUseList);
                }
                throw new RuntimeException("More than 2 Variable Expressions in : " + currentEx + ", class: " + currentEx.getClass().getCanonicalName());
            } else if (useList.size() == 1) {
                return getIDVariableOnlyInStruct(useList.get(0));
            } else {
                throw new RuntimeException("No Variable Expression in : " + currentEx + ", class: " + currentEx.getClass().getCanonicalName());
            }
        } else {
            throw new RuntimeException("No Variable Expression in : " + currentEx + ", class: " + currentEx.getClass().getCanonicalName());
        }
    }

    public static Expression getMemberOnlyInStruct(Expression currentEx) {
        if (currentEx instanceof Identifier) {
            return null;
        } else if (currentEx instanceof AccessExpression) {
            return ((AccessExpression) currentEx).getRHS();
        } else if (currentEx instanceof UnaryExpression) {
            return getMemberOnlyInStruct(((UnaryExpression) currentEx).getExpression());
        } else if (currentEx instanceof ArrayAccess) {
//            traverseIR(currentEx);
            return getMemberOnlyInStruct(((ArrayAccess) currentEx).getArrayName());
        } else if (currentEx instanceof NameID) {
            return null;
        } else {
            throw new RuntimeException("Expression: " + currentEx.toString() + ", class: " + currentEx.getClass().getCanonicalName());
        }
    }

    public static void setKillBitForAlias(
            DFANode cfgNode,
            AnalysisTarget defMapEntry[],
            Expression currentEx,
            Statement currentStmt,
            Procedure proc) {
	        Set<Symbol> definedSymbolsInProc = getDefSymbol(defMapEntry, proc); //DataFlowTools.getDefSymbol(proc);
        Symbol symForCurrentEx = SymbolTools.getSymbolOf(currentEx);
        Domain aliasInfo = IPPointsToAnalysis.getPointsToRelations(currentStmt);
        PointsToDomain aliasInfoForCurrentStmt = null;
        if (aliasInfo instanceof PointsToDomain) {
            aliasInfoForCurrentStmt = (PointsToDomain) aliasInfo;
        }
        if (aliasInfoForCurrentStmt != null) {
            for (Symbol definedSym : definedSymbolsInProc) {   // def symbols for the proc
                Set<PointsToRel> aliasSetAffectingCurrentStmt = aliasInfoForCurrentStmt.get(definedSym);
                if (aliasSetAffectingCurrentStmt != null) {
                    for (PointsToRel aliasInstance : aliasSetAffectingCurrentStmt) {
                        if (symForCurrentEx.getSymbolName().equals(aliasInstance.getPointedToSymbol().getSymbolName()) && aliasInstance.isDefinite()) {
                            // Kill Alias's Definition as well
                            for (int defmapIdx = 0; defmapIdx < defMapEntry.length; defmapIdx++) {
                                Expression idEx = getIDExpression(defMapEntry[defmapIdx].getExpression());
                                if (idEx.toString().equals(aliasInstance.getPointerSymbol().getSymbolName())) {
                                    if (SymbolTools.isPointer(defMapEntry[defmapIdx].getExpression()) == false) {
                                        BitSet killBitSet = cfgNode.getData("KillSet");
                                        killBitSet.set(defmapIdx);
                                        cfgNode.putData("KillSet", killBitSet);
                                    }
                                }
                            }
                        } else if (symForCurrentEx.getSymbolName().equals(aliasInstance.getPointerSymbol().getSymbolName()) && aliasInstance.isDefinite()) {
                            // Kill Alias's Definition as well
                            for (int defmapIdx = 0; defmapIdx < defMapEntry.length; defmapIdx++) {
                                Expression idEx = getIDExpression(defMapEntry[defmapIdx].getExpression());
                                if (idEx.toString().equals(aliasInstance.getPointedToSymbol().getSymbolName())) {
                                    if (SymbolTools.isPointer(defMapEntry[defmapIdx].getExpression()) == false) {
                                        BitSet killBitSet = cfgNode.getData("KillSet");
                                        killBitSet.set(defmapIdx);
                                        cfgNode.putData("KillSet", killBitSet);
                                    }
                                }
                            }
                        }
                    } // for (PointsToRel aliasInstance : aliasSetAffectingCurrentStmt)
                } // if (aliasSetAffectingCurrentStmt != null)
            } // for (Symbol definedSym : definedSymbolsInProc)
        } // if (aliasInfoForCurrentStmt != null)
    }

    public static void setKillBit(DFANode cfgNode, AnalysisTarget defMapEntry[], Expression currentEx, Procedure proc) {
        Expression killEx = currentEx;
        if (currentEx instanceof ConditionalExpression) {
            ConditionalExpression condEx = (ConditionalExpression) currentEx;
            Expression trueEx = condEx.getTrueExpression();
            Expression falseEx = condEx.getFalseExpression();
            if (isStructureAccess(trueEx, proc) && isStructureAccess(falseEx, proc)) {
                Expression idOnlyExTrue = ChainTools.getIDVariableOnlyInStruct(trueEx);
                Expression idOnlyExFalse = ChainTools.getIDVariableOnlyInStruct(falseEx);
                if (idOnlyExTrue != null && idOnlyExFalse != null && idOnlyExTrue.equals(idOnlyExFalse)) {
                    killEx = idOnlyExTrue;
                } else {
                    return;
                }
            } else if (isArrayAccess(trueEx) && isArrayAccess(falseEx)) {
                Expression idOnlyExTrue = ChainTools.getIDExpression(trueEx);
                Expression idOnlyExFalse = ChainTools.getIDExpression(falseEx);
                if (idOnlyExTrue != null && idOnlyExFalse != null && idOnlyExTrue.equals(idOnlyExFalse)) {
                    killEx = idOnlyExTrue;
                } else {
                    return;
                }
            } else if (isPointerAccess(trueEx) && isPointerAccess(falseEx)) {
                Expression idOnlyExTrue = ChainTools.getIDExpression(trueEx);
                Expression idOnlyExFalse = ChainTools.getIDExpression(falseEx);
                if (idOnlyExTrue != null && idOnlyExFalse != null && idOnlyExTrue.equals(idOnlyExFalse)) {
                    killEx = idOnlyExTrue;
                } else {
                    return;
                }
            } else {
                if (trueEx.equals(falseEx)) {
                    killEx = trueEx;
                } else {
                    return;
                }
            }
        }
        if (isArrayAccess(killEx)) {
            for (int i = 0; i < defMapEntry.length; i++) {
                if (isArrayAccess(defMapEntry[i].getExpression())) {
                    if (ChainTools.matchArrayAccessForKill(defMapEntry[i].getExpression(), killEx)) {
                        BitSet killBitSet = cfgNode.getData("KillSet");
                        killBitSet.set(i);
                        cfgNode.putData("KillSet", killBitSet);
                    }
                }
            }
        } else if (isStructureAccess(killEx, proc)) {
            for (int i = 0; i < defMapEntry.length; i++) {
                if (isStructureAccess(defMapEntry[i].getExpression(), proc)) {
                    Expression defVariable = getIDVariableOnlyInStruct(defMapEntry[i].getExpression());
                    Expression defMember = getMemberOnlyInStruct(defMapEntry[i].getExpression());
                    Expression currentVariable = getIDVariableOnlyInStruct(killEx);
                    Expression currentMember = getMemberOnlyInStruct(killEx);
                    if (currentMember == null) {
                        // kill all with same defVariable
                        if (hasSameToString(defVariable, currentVariable)) {
                            BitSet killBitSet = cfgNode.getData("KillSet");
                            killBitSet.set(i);
                            cfgNode.putData("KillSet", killBitSet);
                        }
                    } else {
                        if (defMember != null) {
                            if (hasSameToString(defVariable, currentVariable) &&
                                    hasSameToString(defMember, currentMember)) {
                                BitSet killBitSet = cfgNode.getData("KillSet");
                                killBitSet.set(i);
                                cfgNode.putData("KillSet", killBitSet);
                            }
                        }
                    }
                }
            }
        } else if (isPointerAccess(killEx)) {
            for (int i = 0; i < defMapEntry.length; i++) {
                if (defMapEntry[i].getExpression() instanceof UnaryExpression || defMapEntry[i].getExpression() instanceof ArrayAccess) {
                    if (ChainTools.matchPointerExForKill(defMapEntry[i].getExpression(), killEx)) {
                        BitSet killBitSet = cfgNode.getData("KillSet");
                        killBitSet.set(i);
                        cfgNode.putData("KillSet", killBitSet);
                    }
                } else {
                    if (ChainTools.matchExpressionForKill(defMapEntry[i].getExpression(), killEx)) {
                        BitSet killBitSet = cfgNode.getData("KillSet");
                        killBitSet.set(i);
                        cfgNode.putData("KillSet", killBitSet);
                    }
                }
            }
        } else {
            for (int i = 0; i < defMapEntry.length; i++) {
                // Handle the case of the array access
                if (defMapEntry[i].getExpression() instanceof ArrayAccess) {
                    if (ChainTools.matchArrayAccessForKill(defMapEntry[i].getExpression(), killEx)) {
                        BitSet killBitSet = cfgNode.getData("KillSet");
                        killBitSet.set(i);
                        cfgNode.putData("KillSet", killBitSet);
                    }
                } else {
                    // check the exact matching
                    if (ChainTools.matchExpressionForKill(defMapEntry[i].getExpression(), killEx)) {
                        // compare String between current Expression and Def Mapping
                        BitSet killBitSet = cfgNode.getData("KillSet");
                        killBitSet.set(i);
                        cfgNode.putData("KillSet", killBitSet);
                    }
                }
                // Handle the case of pointer dereferencing
                if (defMapEntry[i].getExpression() instanceof UnaryExpression) {
                    if (ChainTools.matchPointerExForKill(defMapEntry[i].getExpression(), killEx)) {
                        BitSet killBitSet = cfgNode.getData("KillSet");
                        killBitSet.set(i);
                        cfgNode.putData("KillSet", killBitSet);
                    }
                }
            }
        }
    }

    public static void setInBit(DFANode cfgNode, int idx) {
        BitSet inBitSet = cfgNode.getData("InSet");
        inBitSet.set(idx);
        cfgNode.putData("InSet", inBitSet);
    }

    public static void setOutBit(DFANode cfgNode, int idx) {
        BitSet outBitSet = cfgNode.getData("OutSet");
        outBitSet.set(idx);
        cfgNode.putData("OutSet", outBitSet);
    }

    public static boolean checkInBitSet(DFANode cfgNode, int idx) {
        BitSet inBitSet = cfgNode.getData("InSet");
        return inBitSet.get(idx);
    }

    public static boolean matchExpressionForGen(Expression defMapEntry, Expression currentEx) {
        if (System.identityHashCode(defMapEntry) == System.identityHashCode(currentEx)) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean matchArrayAccessForKill(Expression defMapEntryExp, Expression currentEx) {
        if (isArrayAccessWithConstantIndex(defMapEntryExp) ||
                isArrayAccessWithConstantIndex(currentEx)) {
            // exact match is required
            return hasSameToString(defMapEntryExp, currentEx);
        }
        return false;
    }

    public static boolean matchExpressionForKill(Expression defMapEntry, Expression currentEx) {
        if (hasSameToString(defMapEntry, currentEx)) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean matchPointerExForKill(Expression defMapEntry, Expression currentEx) {
        Expression id1 = getIDExpression(defMapEntry);
        Expression id2 = getIDExpression(currentEx);
        return hasSameToString(id1, id2);
    }

    public static boolean isGlobal(Expression ex) {
        return SymbolTools.isGlobal(SymbolTools.getSymbolOf(ex));
    }

    private static boolean isSameExpression(Object ex1, Object ex2) {
        return System.identityHashCode(ex1) == System.identityHashCode(ex2);
    }

    private static CFGraph getArrayVariableDeclaratorControlFlowGraph(VariableDeclarator arrayDec) {
        return new CFGraph(arrayDec, true);
    }

    private static CFGraph getArrayVariableNestedDeclaratorControlFlowGraph(NestedDeclarator arrayDec) {
        return new CFGraph(arrayDec, true);
    }

    public static boolean isDefInItself(DFANode defNode, Expression defEx, Expression useEx) {
        VariableDeclarator declarator = (VariableDeclarator) CFGraph.getIR(defNode);
        List<Expression> defList = getDefListInDec(declarator);
        CFGraph cfg = getArrayVariableDeclaratorControlFlowGraph(declarator);
        Iterator cfgIter = cfg.iterator();
        DFANode defFoundNode = null;
        boolean firstDefFound = false;
        while (cfgIter.hasNext()) {
            DFANode currentCFGNode = (DFANode) cfgIter.next();
            Object nodeIR = CFGraph.getIR(currentCFGNode);
            if (nodeIR == null) {
                continue;
            }
            IRIterator decIter = new DepthFirstIterator((Traversable) nodeIR);
            while (decIter.hasNext()) {
                Object decIR = decIter.next();
                if (decIR instanceof Identifier) {
                    if (isSameExpression(decIR, defEx)) {
                        if (firstDefFound == false) {
                            defFoundNode = currentCFGNode;
                            firstDefFound = true;
                        }
                    }
                    if (isSameExpression(decIR, useEx)) {
                        if (firstDefFound && defFoundNode == currentCFGNode) {
                            // def found in the same cfgNode so use uses the previous def
                            if (defList.contains(useEx)) {
                                // def and use entry are same one
                                return true;
                            } else {
                                return false;
                            }
                        } else if (firstDefFound && defFoundNode != currentCFGNode) {
                            // def found in the previous cfgNode so use uses the def defined in the same cfgNode
                            return true;
                        } else {
                            return false;
                        }
                    }
                }
            }
        }
        return false;
    }

    public static boolean isNestedDefInItself(DFANode defNode, Expression defEx, Expression useEx) {
        NestedDeclarator declarator = (NestedDeclarator) CFGraph.getIR(defNode);
        List<Expression> defList = getDefListInNestedDec(declarator);
        CFGraph cfg = getArrayVariableNestedDeclaratorControlFlowGraph(declarator);
        Iterator cfgIter = cfg.iterator();
        DFANode defFoundNode = null;
        boolean firstDefFound = false;
        while (cfgIter.hasNext()) {
            DFANode currentCFGNode = (DFANode) cfgIter.next();
            Object nodeIR = CFGraph.getIR(currentCFGNode);
            if (nodeIR == null) {
                continue;
            }
            IRIterator decIter = new DepthFirstIterator((Traversable) nodeIR);
            while (decIter.hasNext()) {
                Object decIR = decIter.next();
                if (decIR instanceof Identifier) {
                    if (isSameExpression(decIR, defEx)) {
                        if (firstDefFound == false) {
                            defFoundNode = currentCFGNode;
                            firstDefFound = true;
                        }
                    }
                    if (isSameExpression(decIR, useEx)) {
                        if (firstDefFound && defFoundNode == currentCFGNode) {
                            // def found in the same cfgNode so use uses the previous def
                            if (defList.contains(useEx)) {
                                // def and use entry are same one
                                return true;
                            } else {
                                return false;
                            }
                        } else if (firstDefFound && defFoundNode != currentCFGNode) {
                            // def found in the previous cfgNode so use uses the def defined in the same cfgNode
                            return true;
                        } else {
                            return false;
                        }
                    }
                }
            }
        }
        return false;
    }

    public static List<Expression> getSideEffectParamList(List<Expression> paramList, int[] sideEffectIdx) {
        List<Expression> returnList = new ArrayList<Expression>();
        for (int idx : sideEffectIdx) {
            if (idx < paramList.size()) {
                returnList.add(paramList.get(idx));
            }
        }
        return returnList;
    }

    public static boolean matchIdInExpression(Expression ex1, Expression ex2) {
        if (ex1 == null || ex2 == null) {
            return false;
        }
        IDExpression idEx1 = getIDExpression(ex1);
        IDExpression idEx2 = getIDExpression(ex2);
        if (idEx1 == null || idEx1 == null) {
            return false;
        }
        return (idEx1.equals(idEx2));
    }

    public static boolean matchRefIdInExpression(Expression ex1, Expression ex2, Procedure proc) {
        if (ex1 == null || ex2 == null) {
            return false;
        }
        IDExpression idEx1 = getRefIDExpression(ex1, proc);
        IDExpression idEx2 = getRefIDExpression(ex2, proc);
        if (idEx1 == null || idEx1 == null) {
            return false;
        }
        return (idEx1.equals(idEx2));
    }
    
    public static IDExpression getIDExpression(Expression ex) {
        if (ex == null) {
            return null;
        }
        IDExpression returnEx = null;
        if (ex instanceof UnaryExpression) {
            returnEx = getIDExpression(((UnaryExpression) ex).getExpression());
        } else if (ex instanceof Identifier) {
            returnEx = (IDExpression) ex;
        } else if (ex instanceof NameID) {
            returnEx = (IDExpression) ex;
        } else if (ex instanceof ArrayAccess) {
            returnEx = getIDExpression(((ArrayAccess) ex).getArrayName());
        } else if (ex instanceof Literal) {
            return null;
        } else if (ex instanceof AccessExpression) {
            returnEx = getIDExpression(((AccessExpression) ex).getLHS());
        } else if (ex instanceof FunctionCall) {
            returnEx = null;
        } else if (ex instanceof Typecast) {
            returnEx = getIDExpression(((Typecast) ex).getExpression());
        } else if (ex instanceof BinaryExpression) {
            List<Expression> useList = getUseList(ex);
            if (useList.size() > 1) {
//            	List<Expression> pointerList = new ArrayList<Expression>();
//                for (Expression expInUseList : useList) {
//                	if (SymbolTools.isPointer(expInUseList)) {
//                		pointerList.add(expInUseList);
//                	}
//                    System.out.println("expInUseList: " + expInUseList);
//                }
//                if (pointerList.size() == 1) {
//                	return (IDExpression) pointerList.get(0);
//                }
                throw new RuntimeException("getUseList returns more than 2 List in a Use entry: " + ex + ", class: " + ex.getClass().getCanonicalName() + ", proc: " + IRTools.getParentProcedure(ex));
            } else if (useList.size() == 1) {
                returnEx = getIDExpression(useList.get(0));
            } else {
                return null;
            }
        } else {
            String className = null;
            if (ex != null) {
                className = ex.getClass().getCanonicalName();
            }
            throw new RuntimeException("Unknown Type of Expression: " + ex + ", class: " + className + ", proc: " + IRTools.getParentProcedure(ex));
        }
        return returnEx;
    }

    /**
     * Extract the variable ID from the input expression.
     * Additional feature is to extract the ID from "reference_variable + primitive_variable" form.
     * @param ex
     * @param proc 
     * @return
     */
    public static IDExpression getRefIDExpression(Expression ex, Procedure proc) {
        if (ex == null) {
            return null;
        }
        IDExpression returnEx = null;
        if (ex instanceof UnaryExpression) {
            returnEx = getRefIDExpression(((UnaryExpression) ex).getExpression(), proc);
        } else if (ex instanceof Identifier) {
            returnEx = (IDExpression) ex;
        } else if (ex instanceof NameID) {
            returnEx = (IDExpression) ex;
        } else if (ex instanceof ArrayAccess) {
            returnEx = getRefIDExpression(((ArrayAccess) ex).getArrayName(), proc);
        } else if (ex instanceof Literal) {
            return null;
        } else if (ex instanceof AccessExpression) {
            returnEx = getRefIDExpression(((AccessExpression) ex).getLHS(), proc);
        } else if (ex instanceof FunctionCall) {
            returnEx = null;
        } else if (ex instanceof Typecast) {
            returnEx = getRefIDExpression(((Typecast) ex).getExpression(), proc);
        } else if (ex instanceof BinaryExpression) {
        	List<Expression> useList = new ArrayList<Expression>();
        	List<Expression> useListOrig = getUseList(ex);
        	for (Expression e: useListOrig) {
        		// sprintf((visual_tbl+strlen(visual_tbl)), "  %-8d", vis_no_unit[i]->fld[0].rtint);
        		if (useList.contains(e) == false) {
//        			System.out.println("[getRefIDExpression]added Ex: " + e);
        			useList.add(e);
        		}
          	}
            //
            if (useList.size() > 1) {
            	if (countIDEx(ex) == 1) {
            		// there is duplicate (((char * )( & r))+(2*6)) ==> r and &r
            		DepthFirstIterator<Traversable> dfi = new DepthFirstIterator<Traversable>(ex);
            		while(dfi.hasNext()) {
            			Traversable o = dfi.next();
            			if (o instanceof Identifier) {
            				return (Identifier) o;
            			} else if (o instanceof NameID) {
            				return (NameID) o;
            			}
            		}
            	}
            	List<Expression> pointerList = new ArrayList<Expression>();
                for (Expression expInUseList : useList) {
                	if (isPointerAccess(expInUseList)) {
//            			System.out.println("[getRefIDExpression]pointer Ex: " + expInUseList);
                		pointerList.add(expInUseList);
                	} else if (isArrayAccess(expInUseList)) {
//            			System.out.println("[getRefIDExpression]array Ex: " + expInUseList);
                		pointerList.add(expInUseList);
                	} else if (isStructureAccess(expInUseList, proc)) {
//            			System.out.println("[getRefIDExpression]struct Ex: " + expInUseList);
              			Expression memEx = getMemberOnlyInStruct(expInUseList);
              			if (memEx != null) {
	              			if (isPointerAccess(memEx) || isArrayAccess(memEx)) {
//	              				System.out.println("[getRefIDExpression]struct ref memEx: " + memEx);
	              				pointerList.add(expInUseList);
	              			} else {
//	              				System.out.println("[getRefIDExpression]struct not memEx: " + memEx);
	              			}
              			}
                	}
                }
                if (pointerList.size() == 1) {
                	return getRefIDExpression(pointerList.get(0), proc);
                } else {
                	// (df->insns+df->insn_size)
                	List<Expression> exList = new ArrayList<Expression>();
                	for (Expression pEx: pointerList) {
              			Expression memEx = getMemberOnlyInStruct(pEx);
              			if (memEx != null) {
	              			if (isPointerAccess(memEx) || isArrayAccess(memEx)) {
//	              				System.out.println("Add ref struct: " + memEx);
	              				exList.add(getRefIDExpression(pEx, proc));
	              			} else {
//	              				System.out.println("Remove non ref struct: " + memEx);
	              			}
              			} else {
//              				System.out.println("[getRefIDExpression]: memEx[NULL]: " + pEx);
              				traverseIR(pEx);
              			}
                	}
                	if (exList.size() == 1) {
                		return getRefIDExpression(exList.get(0), proc);
                	} else {
                        // (((unsigned char * )permanent_obstack.next_free)-nbytes)
                        // in get_set_constructor_bytes(exp, (((unsigned char * )permanent_obstack.next_free)-nbytes), nbytes);
                		for (Expression eeee: exList) {
//                    		System.out.println("[getRefIDExpresion]Survivors!!");
                    		traverseIR(eeee);
                		}
                	}
                }
                return null;
//                throw new RuntimeException("getUseList returns more than 2 List in a Use entry: " + ex + ", class: " + ex.getClass().getCanonicalName() + ", proc: " + IRTools.getParentProcedure(ex));
            } else if (useList.size() == 1) {
                returnEx = getRefIDExpression(useList.get(0), proc);
            } else {
                return null;
            }
        } else {
            String className = null;
            if (ex != null) {
                className = ex.getClass().getCanonicalName();
            }
            throw new RuntimeException("Unknown Type of Expression: " + ex + ", class: " + className + ", proc: " + IRTools.getParentProcedure(ex));
        }
        return returnEx;
    }
    
    public static Set<Procedure> getProcedureSet(Program program) {
        List<Procedure> pList = IRTools.getProcedureList(program);
        Set<Procedure> resultProcSet = new LinkedHashSet<Procedure>();
        for (Procedure proc : pList) {
            String sName = proc.getSymbolName();
            if (sName.equals("gnu_dev_major") ||
                    sName.equals("gnu_dev_minor") ||
                    sName.equals("gnu_dev_makedev")) {
                // gcc creates these procs so do not add to the procedure set
            } else {
                resultProcSet.add(proc);
            }
        }
        return resultProcSet;
    }

    public static boolean isEntryNode(DFANode node, CFGraph cfg) {
//        DFANode entryNode = cfg.getEntry();
//        if (entryNode.equals(node)) {
//            return true;
//        } else {
//            return false;
//        }
        if (node.getPreds().size() == 0 && node.getSuccs().size() > 0) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean isExitNode(DFANode node, CFGraph cfg) {
//        List<DFANode> exitNodeList = cfg.getExitNodes();
//        if (exitNodeList.contains(node)) {
//            return true;
//        } else {
//            return false;
//        }
        if (node.getSuccs().size() == 0 && node.getPreds().size() > 0) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean isIdentical(Object obj1, Object obj2) {
        return System.identityHashCode(obj1) == System.identityHashCode(obj2);
    }

    public static List<Expression> getUseList(Traversable traversable) {
//        System.out.println("[ChainTools.getUseList]traversable: " + traversable);
        List<Expression> retList = new ArrayList<Expression>();
        LinkedList<Expression> workList = new LinkedList<Expression>();
        workList.addAll(DataFlowTools.getUseList(traversable));
        while (workList.isEmpty() == false) {
            Expression currentUse = workList.remove();
            if (currentUse instanceof UnaryExpression) {
                workList.addAll(DataFlowTools.getUseList(((UnaryExpression) currentUse).getExpression()));
            } else if (currentUse instanceof AccessExpression) {
                // (t=(tl+(((unsigned)b)&ml)))->e
                if (countIDEx(((AccessExpression) currentUse).getLHS()) > 1) {
                    workList.addAll(DataFlowTools.getUseList(((AccessExpression) currentUse).getLHS()));
                } else if (containsIDEx(currentUse)) {
                    retList.add(currentUse);
                }
            } else if (currentUse instanceof ArrayAccess) {
                // ((inbuf+inptr)+2)[1]
                if (countIDEx(((ArrayAccess) currentUse).getArrayName()) > 1) {
                    workList.addAll(DataFlowTools.getUseList(((ArrayAccess) currentUse).getArrayName()));
                } else if (containsIDEx(currentUse)) {
                    retList.add(currentUse);
                }
            } else if (currentUse instanceof AssignmentExpression) {
                // *(t = &(q->v.t)) = (struct huft *)NULL;
                // (t = &(q->v.t)): AssignmentExpression)
                workList.addAll(DataFlowTools.getUseList(currentUse));
            } else {
                if (containsIDEx(currentUse)) {
                    retList.add(currentUse);
                }
//                System.out.println("[ChainTools.getUseList]add use: " + currentUse + ", class: " + currentUse.getClass().getCanonicalName());
            }
        }
        return retList;
    }

    static int countIDEx(Traversable tr) {
        int cnt = 0;
        DepthFirstIterator iter = new DepthFirstIterator(tr);
        while (iter.hasNext()) {
            Object obj = iter.next();
            if (obj instanceof Identifier) {
                cnt++;
            } else if (obj instanceof NameID) {
                cnt++;
            }
        }
        return cnt;
    }

    static boolean containsIDEx(Traversable tr) {
        DepthFirstIterator iter = new DepthFirstIterator(tr);
        while (iter.hasNext()) {
            Object obj = iter.next();
            if (obj instanceof Identifier) {
                return true;
            } else if (obj instanceof NameID) {
                return true;
            }
        }
        return false;
    }

    public static Set<Symbol> getDefSymbol(Set<AnalysisTarget> defSet, Procedure proc) {
        Set<Symbol> defSymSet = new HashSet<Symbol>();
        for (AnalysisTarget def : defSet) {
            Symbol sym = SymbolTools.getSymbolOf(def.getExpression());
            if (sym == null) {
                if (def.getDFANode().getData("ir") instanceof VariableDeclarator) {
                    defSymSet.add((Symbol) def.getDFANode().getData("ir"));
                } else if (def.getDFANode().getData("ir") instanceof NestedDeclarator) {
                    defSymSet.add((Symbol) def.getDFANode().getData("ir"));
                } else {
                    VariableDeclaration param = def.getDFANode().getData("param");
                    if (param != null) {
                        defSymSet.add((Symbol) param.getDeclarator(0));
                    } else {
//                        System.out.println("def.getDFANode: " + def.getDFANode().getData("ir").getClass().getCanonicalName());
//                        throw new RuntimeException("Cannot find symbol: " + def.getExpression() + ", IR: " + CFGraph.getIR(def.getDFANode()) + ", proc: " + proc.getSymbolName());
//                        System.out.println("[Warning]Cannot find symbol: " + def.getExpression() + ", IR: " + CFGraph.getIR(def.getDFANode()) + ", proc: " + proc.getSymbolName());
                    }
                }
            } else {
                defSymSet.add(sym);
            }
        }
        return defSymSet;
    }

    public static Set<Symbol> getDefSymbol(AnalysisTarget[] defArray, Procedure proc) {
        Set<AnalysisTarget> defSet = new HashSet<AnalysisTarget>();
        for (AnalysisTarget def : defArray) {
//            System.out.println("[getDefSymbol]def: " + def.getExpression());
            defSet.add(def);
        }
        return getDefSymbol(defSet, proc);
    }

    /**
     * Extract the variable ID from the input expression.
     * Additional feature is to extract the ID from "reference_variable + primitive_variable" form.
     * @param ex
     * @param proc 
     * @return
     */
//    public static Expression getVarIDInMixedForm(Expression ex, Procedure proc) {
//        if (ex == null) {
//            return null;
//        }
//        Expression returnEx = null;
//        if (ex instanceof UnaryExpression) {
//            returnEx = getVarIDInMixedForm(((UnaryExpression) ex).getExpression(), proc);
//        } else if (ex instanceof Identifier) {
//            returnEx = (IDExpression) ex;
//        } else if (ex instanceof NameID) {
//            returnEx = (IDExpression) ex;
//        } else if (ex instanceof ArrayAccess) {
//            returnEx = getVarIDInMixedForm(((ArrayAccess) ex).getArrayName(), proc);
//        } else if (ex instanceof Literal) {
//            return null;
//        } else if (ex instanceof AccessExpression) {
//            returnEx = getVarIDInMixedForm(((AccessExpression) ex).getLHS(), proc);
//        } else if (ex instanceof FunctionCall) {
//            returnEx = null;
//        } else if (ex instanceof Typecast) {
//            returnEx = getVarIDInMixedForm(((Typecast) ex).getExpression(), proc);
//        } else if (ex instanceof BinaryExpression) {
//            List<Expression> useList = getUseList(ex);
//            Expression pEx = null;
//            if (useList.size() > 1) {
//                for (Expression use : useList) {
//                    if (getVarIDInMixedForm(use, proc) != null) {
//                        if (ChainTools.isStructureAccess(use, proc)) {
//                            if (ChainTools.isPointerAccess(getMemberOnlyInStruct(use))) {
//                                if (pEx == null || pEx.equals(use)) {
//                                    pEx = use;
//                                } else {
//                                    traverseIR(pEx);
//                                    throw new RuntimeException("Two pointer variable in one argument: previous: " + pEx + ", current: " + use + ", mixedParam: " + ex);
//                                }
//                            }
//                        } else if (ChainTools.isPointerAccess(use)) {
//                            if (pEx == null || pEx.equals(use)) {
//                                pEx = use;
//                            } else {
//                                traverseIR(pEx);
//                                throw new RuntimeException("Two pointer variable in one argument: previous: " + pEx + ", current: " + use + ", mixedParam: " + ex);
//                            }
//                        }
//                    }
//                }
//                returnEx = getVarIDInMixedForm(pEx, proc);
//            } else if (useList.size() == 1) {
//                returnEx = getVarIDInMixedForm(useList.get(0), proc);
//            } else {
//                return null;
//            }
//        } else {
//            String className = null;
//            if (ex != null) {
//                className = ex.getClass().getCanonicalName();
//            }
//            throw new RuntimeException("Unknown Type of Expression: " + ex + ", class: " + className + ", proc: " + IRTools.getParentProcedure(ex));
//        }
//        return returnEx;
//    }

    /**
     * This returns the definition list in the plain assignment statement;
     * @param tr
     * @return
     */
    public static List<Expression> getPlainDefList(Traversable tr, Procedure proc) {
        List<Expression> retList = new ArrayList<Expression>();
        LinkedList<Expression> workList = new LinkedList<Expression>();
        workList.addAll(DataFlowTools.getDefList(tr));
        while (workList.isEmpty() == false) {
            Expression def = workList.remove();
            if (def instanceof UnaryExpression) {
                Expression ex = ((UnaryExpression) def).getExpression();
                if (ex instanceof AssignmentExpression) {
                    // *(t = &(q->v.t)) = (struct huft *)NULL;
                    workList.addAll(DataFlowTools.getDefList(ex));
                } else {
                    if (IRTools.containsFunctionCall(def)) {
                        // ( * __errno_location())=0;
                    } else {
                        retList.add(def);
                    }
                }
            } else if (def instanceof AccessExpression) {
            	// gcc: ((test_bb->succ->flags&1) ? test_bb->succ->succ_next : test_bb->succ)->probability=((test_bb->succ->flags&1) ? test_bb->succ : test_bb->succ->succ_next)->probability;
            	Expression ex = ((AccessExpression)def).getLHS();
            	if (IRTools.containsFunctionCall(ex)) {
                	// (((( * node)->decl.assembler_name!=((tree)((void * )0))) ? ((void)0) : ( * lang_set_decl_assembler_name)(( * node))), ( * node)->decl.assembler_name)->common.static_flag=(( * node)->common.used_flag=1);
            	} else {
	            	if (ex instanceof ConditionalExpression) {
	            		Expression trueEx = ((ConditionalExpression)ex).getTrueExpression();
	            		retList.add(trueEx);
	            		Expression falseEx = ((ConditionalExpression)ex).getFalseExpression();
	            		retList.add(falseEx);
	            	} else {
	            		retList.add(def);
	            	}
            	}
            } else if (def instanceof ConditionalExpression) {
            	// (i ? i: j) = 3;
            	workList.add(((ConditionalExpression)def).getTrueExpression());
            	workList.add(((ConditionalExpression)def).getFalseExpression());
            } else if (def instanceof BinaryExpression) {
            	// (data + j) = 0; // data is ref variable
            	IDExpression idDef = getRefIDExpression(def, proc);
            	if (idDef != null) {
            		retList.add(idDef);
            	} else {
            		throw new RuntimeException("Doesnot have a def in def statement: " + tr);
            	} 
            } else {
                if (IRTools.containsFunctionCall(def)) {
                    // ( * __errno_location())=0;
                } else {
                    retList.add(def);
                }
            }
        }
        return retList;
    }
}

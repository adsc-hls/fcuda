package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.CallGraph;
import cetus.analysis.CallGraph.Node;
import cetus.analysis.DFANode;
import cetus.hir.BinaryExpression;
import cetus.hir.ConditionalExpression;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.StandardLibrary;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * This class performs the creation of program summary graph for executing
 * the following interprocedural data flow analysis.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class ProgramSummaryGraph {

    private Program program;
    private Map<Procedure, CFGraph> cfgMap;
    private final Set<Procedure> procList;
    private Map<Procedure, Set<AnalysisTarget>> refParamMap;
    private Map<AnalysisTarget, Integer> refParamIdxMap;
    private Set<Procedure> refProcSet;
    private Map<String, DFANode> accessIdxMap;  // index for direct access to PSG cfgNode
    private Set<Symbol> globalSet;              // global variables defined or used
    private Set<Symbol> globalSetDeclared;		// global variables declared
    private Set<Procedure> globalDefProcSet;    // proc set containing definitions of global variables
    private Set<AnalysisTarget> globalDefList;  // def list of global variables
    private Set<Procedure> globalUseProcSet;    // proc set containing uses of global variables
    private Set<AnalysisTarget> globalUseList;  // use list of global variables
    private Set<Procedure> globalProcSet;       // proc set directly or indirectly use/def global variables
    private Set<DFANode> workListUniverseRef;
    private Set<DFANode> workListUniverseGlobal;

    public ProgramSummaryGraph(Program program) {
        super();
        this.program = program;
        procList = ChainTools.getProcedureSet(program);
        refParamMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        refParamIdxMap = new HashMap<AnalysisTarget, Integer>();
        refProcSet = new HashSet<Procedure>();
        accessIdxMap = new HashMap<String, DFANode>();
        globalSet = new HashSet<Symbol>();
        globalSetDeclared = new HashSet<Symbol>();
        globalDefList = new LinkedHashSet<AnalysisTarget>();
        globalDefProcSet = new HashSet<Procedure>();
        globalUseList = new LinkedHashSet<AnalysisTarget>();
        globalUseProcSet = new HashSet<Procedure>();
        globalProcSet = new HashSet<Procedure>();
        workListUniverseRef = new LinkedHashSet<DFANode>();
        workListUniverseGlobal = new LinkedHashSet<DFANode>();
    }

    private void generateReferenceParamInfo() {
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> refParamList = new HashSet<AnalysisTarget>();
            Iterator<DFANode> cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode currentNode = cfgIter.next();
                Object currentIR = CFGraph.getIR(currentNode);
                if (currentIR == null) {
                    continue;
                }

                VariableDeclaration param = currentNode.getData("param");
                if (param != null && ChainTools.isNonConstReferenceTypeParameter(param)) {
                    IDExpression idEx = param.getDeclarator(0).getID();
                    AnalysisTarget target = new AnalysisTarget(idEx, currentNode, proc);
                    refParamList.add(target);
                    refParamIdxMap.put(target, (Integer) currentNode.getData("param_idx"));
                }
            }
            if (refParamList.size() == 0) {
                continue;
            } else {
                refParamMap.put(proc, refParamList);
                refProcSet.add(proc);
            }
        }
    }

    private void generateGlobalVariableInfo() {
        List<Traversable> transUnits = program.getChildren();
        for (Traversable tr : transUnits) {
            Set<Symbol> syms = SymbolTools.getGlobalSymbols(tr);
            for (Symbol s : syms) {
                List types = s.getTypeSpecifiers();
                boolean mustInclude = true;
                for (Object o : types) {
                    if (o.toString().equals("typedef")) {
                        mustInclude = false;
                    }
                    if (o.toString().startsWith("enum ")) {
                        mustInclude = false;
                    }
                }
                if (mustInclude) {
                    globalSetDeclared.add(s);
                }
            }
        }
//        for (Symbol s : globalSet) {
//            System.out.println("Global Variable: " + s.getSymbolName());
//        }
    }

    private void generateGlobalDefInfo() {
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("[generatingGlobalDefInfo]Started...");
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode currentNode = (DFANode) cfgIter.next();
                Object currentIR = CFGraph.getIR(currentNode);
                if (currentIR == null) {
                    continue;
                }
                if (currentIR instanceof SwitchStatement) {
                    // do nothing
                } else {
                    List<Expression> defList = ChainTools.getPlainDefList((Traversable) currentIR, proc);//DataFlowTools.getDefList((Traversable) currentIR);
                    for (Expression defEx : defList) {
                        Symbol s = SymbolTools.getSymbolOf(defEx);
                        if (globalSetDeclared.contains(s)) {
                            globalDefList.add(new AnalysisTarget(defEx, currentNode, proc));
                            globalDefProcSet.add(proc);
                            globalSet.add(s);
                        }
                    }
                    List<FunctionCall> fcList = IRTools.getFunctionCalls((Traversable) currentIR);
                    for (FunctionCall fc : fcList) {
                        if (StandardLibrary.hasSideEffectOnParameter(fc)) {
                            int[] paramIndices = StandardLibrary.getSideEffectParamIndices(fc);
                            if (paramIndices == null) {
                                throw new RuntimeException("StandardLibrary.getSideEffectParamIndices(fc) is null: fc: " + fc.getName() + ", please add this information.");
                            }
                            List<Expression> sideEffectList =
                                    ChainTools.getSideEffectParamList(
                                    fc.getArguments(),
                                    paramIndices);
                            for (Expression defEx : sideEffectList) {
                                Symbol s = SymbolTools.getSymbolOf(defEx);
                                if (globalSetDeclared.contains(s)) {
                                    globalDefList.add(new AnalysisTarget(defEx, currentNode, proc));
                                    globalDefProcSet.add(proc);
                                    globalSet.add(s);
                                }
                            }
                        }
                    }
                }
            }
        }
//        for (AnalysisTarget defEx : globalDefList) {
//            System.out.println("global def: " + defEx.getExpression() + ", proc: " + IRTools.getParentProcedure(defEx.getExpression()).getSymbolName());
//        }
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("[generatingGlobalDefInfo]End... globalSet.size: " + globalSet.size() + ", globalSetDeclared.size: " + globalSetDeclared.size());
    }

    private void generateGlobalUseInfo() {
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("[generateGlobalUseInfo]Started...");
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode currentNode = (DFANode) cfgIter.next();
                Object currentIR = CFGraph.getIR(currentNode);
                if (currentIR == null) {
                    continue;
                }
                if (currentIR instanceof SwitchStatement) {
                    List<Expression> useList = ChainTools.getUseList((Traversable) ((SwitchStatement) currentIR).getExpression());
                    for (Expression useEx : useList) {
                        if (ChainTools.getIDExpression(useEx) != null) {
                            Symbol s = SymbolTools.getSymbolOf(useEx);
                            if (globalSetDeclared.contains(s)) {
                                globalUseList.add(new AnalysisTarget(useEx, currentNode, proc));
                                globalUseProcSet.add(proc);
                                globalSet.add(s);
                            }
                        }
                    }
                } else {
                    List<Expression> useList = ChainTools.getUseList((Traversable) currentIR);
                    for (Expression useEx : useList) {
                        if (ChainTools.getIDExpression(useEx) != null) {
                            Symbol s = SymbolTools.getSymbolOf(useEx);
                            if (globalSetDeclared.contains(s)) {
                                globalUseList.add(new AnalysisTarget(useEx, currentNode, proc));
                                globalUseProcSet.add(proc);
                                globalSet.add(s);
                            }
                        }
                    }
                }
            }
        }
//        int idx = 0;
//        for (AnalysisTarget useEx : globalUseList) {
//            System.out.println("global use[" + idx++ + "]: " + useEx.getExpression() + ", proc: " + IRTools.getParentProcedure(useEx.getExpression()).getSymbolName());
//        }
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("[generateGlobalUseInfo]End... globalSet.size: " + globalSet.size());
    }

    private void generateGlobalProcSet() {
        CallGraph cg = new CallGraph(program);
        HashMap cgMap = cg.getCallGraph();
        globalProcSet.addAll(globalDefProcSet);
        globalProcSet.addAll(globalUseProcSet);
        globalProcSet.addAll(refParamMap.keySet());
        Set<Procedure> pList = ChainTools.getProcedureSet(program);
        pList.removeAll(globalProcSet);
        boolean change = true;
        while (change) {
            change = false;
            for (Procedure proc : pList) {
                Node node = (Node) cgMap.get(proc);
                List<Procedure> callees = node.getCallees();
                for (Procedure callee : callees) {
                    if (globalProcSet.contains(callee)) {
                        if (globalProcSet.contains(proc) == false) {
                            globalProcSet.add(proc);
                            change = true;
                        }
                    }
                }
            }
        }
//        System.out.println("Global Proc Set");
//        for (Procedure proc : globalProcSet) {
//            System.out.println("proc: " + proc.getSymbolName());
//        }
    }

    public void buildGraph(Map<Procedure, CFGraph> cfgMap) {
        this.cfgMap = cfgMap;
        // collect information
        generateGlobalVariableInfo();
        generateGlobalDefInfo();
        generateGlobalUseInfo();
        generateReferenceParamInfo();
        generateGlobalProcSet();
        // add cfgNode to CFG
        addRefParamNodes();
        addGlobalNodes();
        //
//        generateDefUseData();
        generateDefUseDataWithThreads();
//        printPSGSummaryDetail();
        propagateDefUseData();
//        printPSGPropagatedDetail();
    }

    // entry & exit node hash key: cfg_node, psg_type(entry or exit), param(ref parameter)
    // call  & ret  node hash key: cfg_node, proc(callee), psg_type(call or return), arg
    private void addRefParamNodes() {
        // add entry and exit cfgNode
        for (Procedure proc : refProcSet) {
            Set<AnalysisTarget> refParamList = refParamMap.get(proc);
            CFGraph cfg = cfgMap.get(proc);
            DFANode node = cfg.getEntry();
            // entry cfgNode
            Set<DFANode> entryList = new HashSet<DFANode>();
            for (AnalysisTarget refParam : refParamList) {
                DFANode entryNode = new DFANode();
                entryNode.putData("cfg_node", node);
                entryNode.putData("psg_type", "entry");
                entryNode.putData("param", refParam);
                entryList.add(entryNode);
                accessIdxMap.put("entry_" + proc.getSymbolName() + "_" + refParamIdxMap.get(refParam).intValue(), entryNode);
            }
            node.putData("psg_entry_ref", entryList);
            // exit cfgNode
            List<DFANode> exitNodeList = cfg.getExitNodes();
            int idx = 0;
            for (DFANode dfaNode : exitNodeList) {
                Set<DFANode> exitList = new HashSet<DFANode>();
                for (AnalysisTarget refParam : refParamList) {
                    DFANode exitNode = new DFANode();
                    exitNode.putData("cfg_node", dfaNode);
                    exitNode.putData("psg_type", "exit");
                    exitNode.putData("param", refParam);
                    exitList.add(exitNode);
                    accessIdxMap.put("exit_" + proc.getSymbolName() + idx + "_" + refParamIdxMap.get(refParam).intValue(), exitNode);
                }
                dfaNode.putData("psg_exit_ref", exitList);
                idx++;
            }
        }

        // add call and return cfgNode (must check all the procs)
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Iterator<DFANode> cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode node = cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(node);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> callList = new HashSet<DFANode>();
                Set<DFANode> returnList = new HashSet<DFANode>();
                for (FunctionCall fc : fcList) {
                    Procedure callee = fc.getProcedure();
                    if (callee == null) {
                        continue;
                    }
                    if (refProcSet.contains(callee) == false) {
                        continue;
                    }
                    Set<AnalysisTarget> argSet = refParamMap.get(callee);
                    for (AnalysisTarget arg : argSet) {
                        // call
                        DFANode callNode = new DFANode();
                        callNode.putData("cfg_node", node);
                        callNode.putData("proc", callee);
                        callNode.putData("psg_type", "call");
                        // handle the case of ConditionalExpression
                        Expression argCallee = fc.getArgument(refParamIdxMap.get(arg).intValue());
                        List<Expression> argsList = new ArrayList<Expression>();
                        if (argCallee instanceof ConditionalExpression) {
                        	argsList = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)argCallee, proc);
                        } else {
                        	argsList.add(argCallee);
                        }
                    	callNode.putData("arg", argsList);
                        callNode.putData("arg_idx", refParamIdxMap.get(arg));
                        callList.add(callNode);
                        DFANode entryNode = accessIdxMap.get("entry_" + callee.getSymbolName() + "_" + refParamIdxMap.get(arg).intValue());
                        if (entryNode == null) {
                            throw new RuntimeException("No Entry Node found: " + "entry_" + callee.getSymbolName() + "_" + refParamIdxMap.get(arg).intValue());
                        }
                        callNode.addSucc(entryNode);
                        entryNode.addPred(callNode);
                        // return
                        DFANode returnNode = new DFANode();
                        returnNode.putData("cfg_node", node);
                        returnNode.putData("proc", callee);
                        returnNode.putData("psg_type", "return");
                        returnNode.putData("arg", argsList);
                        returnNode.putData("arg_idx", refParamIdxMap.get(arg));
                        returnList.add(returnNode);
                        CFGraph cfgCallee = cfgMap.get(callee);
                        List<DFANode> calleeExitList = cfgCallee.getExitNodes();
                        for (int idx = 0; idx < calleeExitList.size(); idx++) {
                            DFANode exitNode = accessIdxMap.get("exit_" + callee.getSymbolName() + idx + "_" + refParamIdxMap.get(arg).intValue());
                            if (exitNode == null) {
                                throw new RuntimeException("No Exit Node found: " + "exit_" + callee.getSymbolName() + "_" + refParamIdxMap.get(arg).intValue());
                            }
                            returnNode.addPred(exitNode);
                            exitNode.addSucc(returnNode);
                        }
                    }
                }
                if (callList.size() > 0) {
                    node.putData("psg_call_ref", callList);
                }
                if (returnList.size() > 0) {
                    node.putData("psg_return_ref", returnList);
                }
            }
        }
    }

    private void addGlobalNodes() {
        // add entry and exit cfgNode
        for (Procedure proc : globalProcSet) {
            CFGraph cfg = cfgMap.get(proc);
//            System.out.println("ExitNode Size: " + cfg.getExitNodes().size() + ", proc: " + proc.getSymbolName());
            DFANode node = cfg.getEntry();
            // entry cfgNode
            DFANode entryNode = new DFANode();
            entryNode.putData("cfg_node", node);
            entryNode.putData("psg_type", "entry");
            accessIdxMap.put("entry_" + proc.getSymbolName(), entryNode);
            node.putData("psg_entry_global", entryNode);
            // exit cfgNode
            List<DFANode> exitNodeList = cfg.getExitNodes();
            int idx = 0;
            for (DFANode dfaNode : exitNodeList) {
                DFANode exitNode = new DFANode();
                exitNode.putData("cfg_node", dfaNode);
                exitNode.putData("psg_type", "exit");
                accessIdxMap.put("exit_" + proc.getSymbolName() + idx++, exitNode);
                dfaNode.putData("psg_exit_global", exitNode);
            }
        }

        // add call and return cfgNode (must check all the procs)
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode node = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(node);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> callList = new HashSet<DFANode>();
                Set<DFANode> returnList = new HashSet<DFANode>();
                for (FunctionCall fc : fcList) {
                    Procedure callee = fc.getProcedure();
                    if (callee == null) {
                        continue;
                    }
                    if (globalProcSet.contains(callee) == false) {
                        continue;
                    }
                    DFANode callNode = new DFANode();
                    callNode.putData("cfg_node", node);
                    callNode.putData("psg_type", "call");
                    callNode.putData("proc", callee);
                    callList.add(callNode);
                    DFANode entryNode = accessIdxMap.get("entry_" + callee.getSymbolName());
                    if (entryNode == null) {
                        throw new RuntimeException("No Entry Node found: " + "entry_" + callee.getSymbolName());
                    }
                    callNode.addSucc(entryNode);
                    entryNode.addPred(callNode);
//                    System.out.println("[" + proc.getSymbolName() + "]added call_node: " + currentCallNode);
                    // return
                    DFANode returnNode = new DFANode();
                    returnNode.putData("cfg_node", node);
                    returnNode.putData("psg_type", "return");
                    returnNode.putData("proc", callee);
                    returnList.add(returnNode);
                    CFGraph cfgCallee = cfgMap.get(callee);
                    List<DFANode> calleeExitList = cfgCallee.getExitNodes();
                    for (int idx = 0; idx < calleeExitList.size(); idx++) {
                        DFANode exitNode = accessIdxMap.get("exit_" + callee.getSymbolName() + idx);
                        if (exitNode == null) {
                            throw new RuntimeException("No Exit Node found: " + "exit_" + callee.getSymbolName());
                        }
                        returnNode.addPred(exitNode);
                        exitNode.addSucc(returnNode);
                    }
//                    System.out.println("[" + proc.getSymbolName() + "]added return_node: " + currentCallNode);
                }
                if (callList.size() > 0) {
                    node.putData("psg_call_global", callList);
                }
                if (returnList.size() > 0) {
                    node.putData("psg_return_global", returnList);
                }
            }
        }
    }

    private void generateDefUseDataWithThreads() {
        Map<Procedure, Set<AnalysisTarget>> targetListMap = null;
        
        // For Reference Parameters
        
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Reaching Definition Analysis for Reference Parameters");
        Set<Procedure> procList = ChainTools.getProcedureSet(program);
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("##### Number of Procs: " + procList.size());
        // start of ThreadVersion
        targetListMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        ExecutorService taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            PSGReachingDefinitionAnalysis rdaRefParam = new PSGReachingDefinitionAnalysis(program, cfgMap, refParamMap, refParamIdxMap);
        	rdaRefParam.setAnalysisTarget(proc, targetListMap);
        	taskExecutor.execute(rdaRefParam);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion

        setRefDefInfoToPSG(targetListMap);

    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Upwards Exposed Uses Analysis for Reference Parameters");
        // PSG Upwards Exposed Uses Analysis for Reference Parameters
        // start of ThreadVersion
        targetListMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            PSGUpwardsExposedUsesAnalysis ueuaRefParam = new PSGUpwardsExposedUsesAnalysis(program, cfgMap, refParamMap, refParamIdxMap);
            ueuaRefParam.setAnalysisTarget(proc, targetListMap);
        	taskExecutor.execute(ueuaRefParam);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion

        setRefUseInfoToPSG(targetListMap);

        // PSG Reaching Definition Analysis for Reference Parameters
        // start of ThreadVersion
        targetListMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            PSGReachingDefinitionAnalysis rdaRefParam = new PSGReachingDefinitionAnalysis(program, cfgMap, refParamMap, refParamIdxMap);
            rdaRefParam.setAnalysisTarget(proc, targetListMap);
        	taskExecutor.execute(rdaRefParam);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion

        setRefDefInfoToPSG(targetListMap);
        addReachingEdgeForRef(targetListMap);
        collectWorkListUniverseRef();
        addInterReachingEdgeForRef(targetListMap);
//        printPSGForRef();
        System.gc();

        // For Global Variables

    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Reaching Definition Analysis for Global Variables");
        
        // PSG Reaching Definition Analysis for Global Variables
        // start of ThreadVersion
        targetListMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            PSGReachingDefinitionAnalysis rdaGlobalParam = new PSGReachingDefinitionAnalysis(program, cfgMap, globalSet, globalDefList, globalDefProcSet);
            rdaGlobalParam.setAnalysisTarget(proc, targetListMap);
        	taskExecutor.execute(rdaGlobalParam);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion

        setGlobalDefInfoToPSG(targetListMap);

        taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            PSGRegionalKillSetAnlaysis rksaGlobalParam = new PSGRegionalKillSetAnlaysis(program, cfgMap, targetListMap, globalProcSet);
            rksaGlobalParam.setAnalysisTarget(proc, targetListMap);
        	taskExecutor.execute(rksaGlobalParam);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion

        setGlobalKillSetInfoToPSG(targetListMap);

    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Upwards Exposed Uses Analysis for Global Variables");
        
        // PSG Upwards Exposed Uses Analysis for Global Variables
        // start of ThreadVersion
        targetListMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            PSGUpwardsExposedUsesAnalysis ueuaGlobalParam = new PSGUpwardsExposedUsesAnalysis(program, cfgMap, globalSet, globalUseList, globalUseProcSet);
            ueuaGlobalParam.setAnalysisTarget(proc, targetListMap);
        	taskExecutor.execute(ueuaGlobalParam);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion

        setGlobalUseInfoToPSG(targetListMap);

        // start of ThreadVersion
        targetListMap = new HashMap<Procedure, Set<AnalysisTarget>>();
        taskExecutor = Executors.newFixedThreadPool(procList.size());
        for (Procedure proc: procList) {
            PSGReachingGlobalProcSetAnalysis rgpsa = new PSGReachingGlobalProcSetAnalysis(program, cfgMap, globalProcSet);
            rgpsa.setAnalysisTarget(proc, targetListMap);
        	taskExecutor.execute(rgpsa);
        }
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        // end of ThreadVersion

        addReachingEdgeForGlobal(targetListMap);
        collectWorkListUniverseGlobal();
//        printPSGForGlobal();
    }
    
    private void generateDefUseData() {
        Map<Procedure, Set<AnalysisTarget>> targetListMap = null;
        // For Reference Parameters
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Reaching Definition Analysis for Reference Parameters");
        // PSG Reaching Definition Analysis for Reference Parameters
        PSGReachingDefinitionAnalysis rdaRefParam = new PSGReachingDefinitionAnalysis(program, cfgMap, refParamMap, refParamIdxMap);
        rdaRefParam.start();
        targetListMap = rdaRefParam.getAnalysisTargetListMap();

        setRefDefInfoToPSG(targetListMap);

    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Upwards Exposed Uses Analysis for Reference Parameters");
        // PSG Upwards Exposed Uses Analysis for Reference Parameters
        PSGUpwardsExposedUsesAnalysis ueuaRefParam = new PSGUpwardsExposedUsesAnalysis(program, cfgMap, refParamMap, refParamIdxMap);
        ueuaRefParam.start();
        targetListMap = ueuaRefParam.getAnalysisTargetListMap();

        setRefUseInfoToPSG(targetListMap);

        // PSG Reaching Definition Analysis for Reference Parameters
        rdaRefParam = new PSGReachingDefinitionAnalysis(program, cfgMap, refParamMap, refParamIdxMap);
        rdaRefParam.start();
        targetListMap = rdaRefParam.getAnalysisTargetListMap();

        setRefDefInfoToPSG(targetListMap);
        addReachingEdgeForRef(targetListMap);
        collectWorkListUniverseRef();
        addInterReachingEdgeForRef(targetListMap);
//        printPSGForRef();

        // For Global Variables

    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Reaching Definition Analysis for Global Variables");
        // PSG Reaching Definition Analysis for Global Variables
        PSGReachingDefinitionAnalysis rdaGlobalParam = new PSGReachingDefinitionAnalysis(program, cfgMap, globalSet, globalDefList, globalDefProcSet);
        rdaGlobalParam.start();
        targetListMap = rdaGlobalParam.getAnalysisTargetListMap();

        setGlobalDefInfoToPSG(targetListMap);

        PSGRegionalKillSetAnlaysis rksaGlobalParam = new PSGRegionalKillSetAnlaysis(program, cfgMap, targetListMap, globalProcSet);
        rksaGlobalParam.start();

        setGlobalKillSetInfoToPSG(targetListMap);

    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("### Upwards Exposed Uses Analysis for Global Variables");
        // PSG Upwards Exposed Uses Analysis for Global Variables
        PSGUpwardsExposedUsesAnalysis ueuaGlobalParam = new PSGUpwardsExposedUsesAnalysis(program, cfgMap, globalSet, globalUseList, globalUseProcSet);
        ueuaGlobalParam.start();
        targetListMap = ueuaGlobalParam.getAnalysisTargetListMap();

        setGlobalUseInfoToPSG(targetListMap);

        PSGReachingGlobalProcSetAnalysis rgpsa = new PSGReachingGlobalProcSetAnalysis(program, cfgMap, globalProcSet);
        rgpsa.start();
        targetListMap = rgpsa.getAnalysisTargetListMap();
        addReachingEdgeForGlobal(targetListMap);
        collectWorkListUniverseGlobal();
//        printPSGForGlobal();

    }

    private void setRefDefInfoToPSG(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        // set local def information at call
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> callList = cfgNode.getData("psg_call_ref");
                if (callList == null) {
                    continue;
                }
                for (DFANode callNode : callList) {
//                    System.out.println("Call Node: " + ((DFANode)currentCallNode.getData("cfg_node")).getData("ir"));
//                    Expression arg = callNode.getData("arg");
                    List<Expression> argsList = callNode.getData("arg");
                    for (Expression arg: argsList) {
	                    BitSet inSet = cfgNode.getData("InSet");
	                    for (int idx = 0; idx < targetArray.length; idx++) {
	                        AnalysisTarget target = targetArray[idx];
	//                        System.out.println("proc: " + proc.getSymbolName() + ", idx: " + idx + ", target: " + target.getExpression() + ", arg: " + arg);
	                        if (inSet.get(idx)) {
	                            if (arg instanceof ConditionalExpression) {
	                            	List<Expression> defList = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)arg, proc);
                                    for (Expression def : defList) {
//                                        if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//                                            IDExpression idEx = ChainTools.getIDExpression(def);
                                            if (ChainTools.matchRefIdInExpression(def, target.getExpression(), proc)) {
                                                Set<AnalysisTarget> defSet = callNode.getData("def");
                                                if (defSet == null) {
                                                    defSet = new LinkedHashSet<AnalysisTarget>();
                                                    callNode.putData("def", defSet);
                                                }
                                                defSet.add(target);
                                            }
//                                        }
                                    }
	                            	
//	                                Expression trueEx = ((ConditionalExpression) arg).getTrueExpression();
//	                                if (trueEx instanceof BinaryExpression) {
//	                                    List<Expression> defList = ChainTools.getUseList(trueEx);
//	                                    for (Expression def : defList) {
//	                                        if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//	                                            IDExpression idEx = ChainTools.getIDExpression(def);
//	                                            if (ChainTools.matchRefIdInExpression(idEx, target.getExpression(), proc)) {
//	                                                Set<AnalysisTarget> defSet = callNode.getData("def");
//	                                                if (defSet == null) {
//	                                                    defSet = new LinkedHashSet<AnalysisTarget>();
//	                                                    callNode.putData("def", defSet);
//	                                                }
//	                                                defSet.add(target);
//	                                            }
//	                                        }
//	                                    }
//	                                } else {
//	                                    Expression idEx = ChainTools.getIDExpression(trueEx);
//	                                    if (ChainTools.matchRefIdInExpression(idEx, target.getExpression(), proc)) {
//	                                        Set<AnalysisTarget> defSet = callNode.getData("def");
//	                                        if (defSet == null) {
//	                                            defSet = new LinkedHashSet<AnalysisTarget>();
//	                                            callNode.putData("def", defSet);
//	                                        }
//	                                        defSet.add(target);
//	                                    }
//	                                }
//	                                Expression falseEx = ((ConditionalExpression) arg).getFalseExpression();
//	                                if (falseEx instanceof BinaryExpression) {
//	                                    List<Expression> defList = ChainTools.getUseList(falseEx);
//	                                    for (Expression def : defList) {
//	                                        if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//	                                            IDExpression idEx = ChainTools.getIDExpression(def);
//	                                            if (ChainTools.matchRefIdInExpression(idEx, target.getExpression(), proc)) {
//	                                                Set<AnalysisTarget> defSet = callNode.getData("def");
//	                                                if (defSet == null) {
//	                                                    defSet = new LinkedHashSet<AnalysisTarget>();
//	                                                    callNode.putData("def", defSet);
//	                                                }
//	                                                defSet.add(target);
//	                                            }
//	                                        }
//	                                    }
//	                                } else {
//	                                    Expression idEx = ChainTools.getIDExpression(falseEx);
//	                                    if (ChainTools.matchIdInExpression(idEx, target.getExpression())) {
//	                                        Set<AnalysisTarget> defSet = callNode.getData("def");
//	                                        if (defSet == null) {
//	                                            defSet = new LinkedHashSet<AnalysisTarget>();
//	                                            callNode.putData("def", defSet);
//	                                        }
//	                                        defSet.add(target);
//	                                    }
//	                                }
//	                            } else if (arg instanceof BinaryExpression) {
//	                                List<Expression> defList = ChainTools.getUseList(arg);
//	                                for (Expression def : defList) {
//	                                    if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//	                                        IDExpression idEx = ChainTools.getIDExpression(def);
//	                                        if (ChainTools.matchIdInExpression(idEx, target.getExpression())) {
//	                                            Set<AnalysisTarget> defSet = callNode.getData("def");
//	                                            if (defSet == null) {
//	                                                defSet = new LinkedHashSet<AnalysisTarget>();
//	                                                callNode.putData("def", defSet);
//	                                            }
//	                                            defSet.add(target);
//	                                        }
//	                                    }
//	                                }
//	                            } else {
//	                                if (ChainTools.matchIdInExpression(arg, target.getExpression())) {
//	                                    Set<AnalysisTarget> defSet = callNode.getData("def");
//	                                    if (defSet == null) {
//	                                        defSet = new LinkedHashSet<AnalysisTarget>();
//	                                        callNode.putData("def", defSet);
//	                                    }
//	                                    defSet.add(target);
//	                                }
	                            }
	                        }
	                    }
                    }
                }
            }
        }

        // set local def information at exit cfgNode
        for (Procedure proc : refProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    // exit cfgNode
                    Set<DFANode> exitList = cfgNode.getData("psg_exit_ref");
                    if (exitList == null) {
                        if (proc.getSymbolName().equals("main")) {
                            continue;
                        }
                        throw new RuntimeException("No exit node for ref found: " + cfgNode.getData("ir") + ", proc: " + proc.getSymbolName());
                    }
                    for (DFANode exitNode : exitList) {
                        //entryNode.putData("param", exitRefParam);
                        AnalysisTarget refParam = exitNode.getData("param");
                        BitSet inSet = cfgNode.getData("InSet");
                        for (int idx = 0; idx < targetArray.length; idx++) {
                            AnalysisTarget target = targetArray[idx];
                            if (inSet.get(idx)) {
                                if (ChainTools.matchRefIdInExpression(refParam.getExpression(), target.getExpression(), proc)) {
                                    Set<AnalysisTarget> defSet = exitNode.getData("def");
                                    if (defSet == null) {
                                        defSet = new LinkedHashSet<AnalysisTarget>();
                                        exitNode.putData("def", defSet);
                                    }
                                    defSet.add(target);
//                                    System.out.println("[" + proc.getSymbolName() + "][exit]def added: " + target.getExpression() + ", node: " + target.getDFANode().getData("ir") + ", to exit node: " + ((DFANode) exitNode.getData("cfg_node")).getData("ir"));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void setRefUseInfoToPSG(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        // set local use information at return
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> returnList = cfgNode.getData("psg_return_ref");
                if (returnList == null) {
                    continue;
                }
                for (DFANode returnNode : returnList) {
//                    System.out.println("Call Node: " + ((DFANode)currentCallNode.getData("cfg_node")).getData("ir"));
//                    Expression arg = returnNode.getData("arg");
                    List<Expression> argsList = returnNode.getData("arg");
                    for (Expression arg: argsList) {
	                    BitSet outSet = cfgNode.getData("OutSet");
	                    for (int idx = 0; idx < targetArray.length; idx++) {
	                        AnalysisTarget target = targetArray[idx];
	//                        System.out.println("idx: " + idx + ", target: " + target.getExpression() + ", arg: " + arg);
	                        if (outSet.get(idx)) {
	                            if (arg instanceof ConditionalExpression) {
	                            	List<Expression> defList = ChainTools.getRefIDExpressionListInConditionalExpressionArg((ConditionalExpression)arg, proc);
//	                                Expression trueEx = ((ConditionalExpression) arg).getTrueExpression();
//	                                if (trueEx instanceof BinaryExpression) {
//	                                    List<Expression> defList = ChainTools.getUseList(trueEx);
	                                    for (Expression def : defList) {
//	                                        if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//	                                            IDExpression idEx = ChainTools.getIDExpression(def);
	                                            if (ChainTools.matchRefIdInExpression(def, target.getExpression(), proc)) {
	                                                Set<AnalysisTarget> useSet = returnNode.getData("use");
	                                                if (useSet == null) {
	                                                    useSet = new LinkedHashSet<AnalysisTarget>();
	                                                    returnNode.putData("use", useSet);
	                                                }
	                                                useSet.add(target);
	                                            }
//	                                        }
	                                    }
//	                                } else {
//	                                    Expression idEx = ChainTools.getIDExpression(trueEx);
//	                                    if (idEx != null) {
//	                                        if (ChainTools.matchIdInExpression(idEx, target.getExpression())) {
//	                                            Set<AnalysisTarget> useSet = returnNode.getData("use");
//	                                            if (useSet == null) {
//	                                                useSet = new LinkedHashSet<AnalysisTarget>();
//	                                                returnNode.putData("use", useSet);
//	                                            }
//	                                            useSet.add(target);
//	                                        }
//	                                    }
//	                                }
//	                                Expression falseEx = ((ConditionalExpression) arg).getFalseExpression();
//	                                if (falseEx instanceof BinaryExpression) {
//	                                    List<Expression> defList = ChainTools.getUseList(falseEx);
//	                                    for (Expression def : defList) {
//	                                        if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//	                                            IDExpression idEx = ChainTools.getIDExpression(def);
//	                                            if (ChainTools.matchIdInExpression(idEx, target.getExpression())) {
//	                                                Set<AnalysisTarget> useSet = returnNode.getData("use");
//	                                                if (useSet == null) {
//	                                                    useSet = new LinkedHashSet<AnalysisTarget>();
//	                                                    returnNode.putData("use", useSet);
//	                                                }
//	                                                useSet.add(target);
//	                                            }
//	                                        }
//	                                    }
//	                                } else {
//	                                    Expression idEx = ChainTools.getIDExpression(falseEx);
//	                                    if (idEx != null) {
//	                                        if (ChainTools.matchIdInExpression(idEx, target.getExpression())) {
//	                                            Set<AnalysisTarget> useSet = returnNode.getData("use");
//	                                            if (useSet == null) {
//	                                                useSet = new LinkedHashSet<AnalysisTarget>();
//	                                                returnNode.putData("use", useSet);
//	                                            }
//	                                            useSet.add(target);
//	                                        }
//	                                    }
//	                                }
//	                            } else if (arg instanceof BinaryExpression) {
//	                                List<Expression> defList = ChainTools.getUseList(arg);
//	                                for (Expression def : defList) {
//	                                    if (ChainTools.isPointerAccess(def) || ChainTools.isArrayAccess(def)) {
//	                                        IDExpression idEx = ChainTools.getIDExpression(def);
//	                                        if (ChainTools.matchIdInExpression(idEx, target.getExpression())) {
//	                                            Set<AnalysisTarget> useSet = returnNode.getData("use");
//	                                            if (useSet == null) {
//	                                                useSet = new LinkedHashSet<AnalysisTarget>();
//	                                                returnNode.putData("use", useSet);
//	                                            }
//	                                            useSet.add(target);
//	                                        }
//	                                    }
//	                                }
//	                            } else {
//	                                if (ChainTools.matchIdInExpression(arg, target.getExpression())) {
//	                                    Set<AnalysisTarget> useSet = returnNode.getData("use");
//	                                    if (useSet == null) {
//	                                        useSet = new LinkedHashSet<AnalysisTarget>();
//	                                        returnNode.putData("use", useSet);
//	                                    }
//	                                    useSet.add(target);
//	                                }
	                            }
	                        }
	                    }
                    }
                }
            }
        }

        // set local use information at entry
        for (Procedure proc : refProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            DFANode cfgNode = cfg.getEntry();
//            Iterator cfgIter = cfg.iterator();
//            while (cfgIter.hasNext()) {
//                DFANode cfgNode = (DFANode) cfgIter.next();
//                if (ChainTools.isEntryNode(cfgNode)) {
            // entry cfgNode
            Set<DFANode> entryList = cfgNode.getData("psg_entry_ref");
            if (entryList == null) {
                throw new RuntimeException("No entry node for ref found: " + cfgNode + ", proc: " + proc.getSymbolName());
            }
            for (DFANode entryNode : entryList) {
                //entryNode.putData("param", exitRefParam);
                AnalysisTarget refParam = entryNode.getData("param");
                BitSet outSet = cfgNode.getData("OutSet");
                for (int idx = 0; idx < targetArray.length; idx++) {
                    AnalysisTarget target = targetArray[idx];
                    if (outSet.get(idx)) {
//                                System.out.println("[" + proc.getSymbolName() + "][entry-outSet]idx: " + idx + ", target: " + target.getExpression() + ", refParam: " + refParam.getExpression());
                        if (ChainTools.matchRefIdInExpression(refParam.getExpression(), target.getExpression(), proc)) {
                            Set<AnalysisTarget> useSet = entryNode.getData("use");
                            if (useSet == null) {
                                useSet = new LinkedHashSet<AnalysisTarget>();
                                entryNode.putData("use", useSet);
                            }
                            useSet.add(target);
//                                    System.out.println("[" + proc.getSymbolName() + "][entry]use added: " + target.getExpression() + ", node: " + target.getDFANode().getData("ir") + ", to entry node: " + ((DFANode) entryNode.getData("cfg_node")).getData("ir"));
                        }
                    }
                }
            }
//                }
//            }
        }
    }

    private void addReachingEdgeForRef(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        // add reaching edges at call node
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> callList = cfgNode.getData("psg_call_ref");
                if (callList == null) {
                    continue;
                }
                for (DFANode currentCallNode : callList) {
//                    Expression arg = currentCallNode.getData("arg");
                    List<Expression> argsList = currentCallNode.getData("arg");
                    for (Expression arg: argsList) {
	                    BitSet inSet = cfgNode.getData("InSet");
	                    for (int idx = 0; idx < targetArray.length; idx++) {
	                        AnalysisTarget target = targetArray[idx];
	                        if (inSet.get(idx)) {
	                            // reaching from entry
	                            if (target.getDFANode().getData("param") != null) {
	                                if (arg.equals(target.getExpression())) {
	//                                    System.out.println("Same Expression at Call Node: " + arg + " in Proc: " + proc.getSymbolName());
	                                    CFGraph cfgTmp = cfgMap.get(proc);
	                                    DFANode entry = cfgTmp.getEntry();
	                                    // check for entry node
	                                    Set<DFANode> targetEntryList = entry.getData("psg_entry_ref");
	                                    if (targetEntryList != null) {
	                                        for (DFANode targetEntryNode : targetEntryList) {
	                                            AnalysisTarget entryRefParam = targetEntryNode.getData("param");
	                                            if (ChainTools.matchRefIdInExpression(arg, entryRefParam.getExpression(), proc)) {
	                                                currentCallNode.addPred(targetEntryNode);
	                                                targetEntryNode.addSucc(currentCallNode);
	                                            }
	                                        }
	                                    }
	                                }
	                            }
	                            DFANode targetNode = target.getDFANode();
	                            if (currentCallNode.equals(targetNode)) {
	//                                System.out.println("Same Node!!");
	                                continue;
	                            }
	                            // check for return node
	                            Set<DFANode> targetReturnList = targetNode.getData("psg_return_ref");
	                            if (targetReturnList != null) {
	                                for (DFANode targetReturnNode : targetReturnList) {
	                                	List<Expression> argsList2 = targetReturnNode.getData("arg");
	                                	for (Expression previousArg: argsList2) {
//	                                    Expression previousArg = targetReturnNode.getData("arg");
//		                                    System.out.println("[addReachingEdgeForRef]previousArg: " + previousArg + ", class: " + previousArg.getClass().getCanonicalName());
//		                                    System.out.println("[addReachingEdgeForRef]arg: " + arg + ", class: " + previousArg.getClass().getCanonicalName());
		                                    if (arg instanceof ConditionalExpression || previousArg instanceof ConditionalExpression) {
		                                        // Temp : need to be modified
		                                        if (arg.equals(previousArg)) {
		                                            currentCallNode.addPred(targetReturnNode);
		                                            targetReturnNode.addSucc(currentCallNode);
		                                        }
		                                    } else if (arg instanceof BinaryExpression || previousArg instanceof BinaryExpression) {
		                                        // Temp : need to be modified
		                                        if (arg.equals(previousArg)) {
		                                            currentCallNode.addPred(targetReturnNode);
		                                            targetReturnNode.addSucc(currentCallNode);
		                                        }
		                                    } else {
		                                        if (ChainTools.matchRefIdInExpression(arg, previousArg, proc)) {
		                                            currentCallNode.addPred(targetReturnNode);
		                                            targetReturnNode.addSucc(currentCallNode);
		                                        }
		                                    }
	                                	}
	                                }
	                            }
	                        }
	                    }
                    }
                }
            }
        }

        // add reaching edges at exit node
        for (Procedure proc : refProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    // exit cfgNode
                    Set<DFANode> exitList = cfgNode.getData("psg_exit_ref");
                    if (exitList == null) {
                        if (proc.getSymbolName().equals("main")) {
                            continue;
                        }
                        throw new RuntimeException("No exit node for ref found: " + cfgNode);
                    }
                    for (DFANode currentExitNode : exitList) {
                        //entryNode.putData("param", exitRefParam);
                        AnalysisTarget exitRefParam = currentExitNode.getData("param");
                        BitSet inSet = cfgNode.getData("InSet");
                        for (int idx = 0; idx < targetArray.length; idx++) {
                            AnalysisTarget target = targetArray[idx];
                            if (inSet.get(idx)) {
                                // reaching from entry
                                if (exitRefParam.equals(target)) {
//                                    System.out.println("Reaching from Entry to Exit: " + exitRefParam.getExpression() + " in Proc: " + proc.getSymbolName());
                                    CFGraph cfgTmp = cfgMap.get(proc);
                                    DFANode entry = cfgTmp.getEntry();
                                    // check for entry node
                                    Set<DFANode> targetEntryList = entry.getData("psg_entry_ref");
                                    if (targetEntryList != null) {
                                        for (DFANode targetEntryNode : targetEntryList) {
                                            AnalysisTarget entryRefParam = targetEntryNode.getData("param");
                                            if (ChainTools.matchRefIdInExpression(exitRefParam.getExpression(), entryRefParam.getExpression(), proc)) {
                                                currentExitNode.addPred(targetEntryNode);
                                                targetEntryNode.addSucc(currentExitNode);
                                            }
                                        }
                                    }
                                }
                                DFANode targetNode = target.getDFANode();
                                // check for return node
                                Set<DFANode> targetReturnList = targetNode.getData("psg_return_ref");
                                if (targetReturnList != null) {
                                    for (DFANode targetReturnNode : targetReturnList) {
                                    	List<Expression> argsList = targetReturnNode.getData("arg");
//                                        Expression previousArg = targetReturnNode.getData("arg");
                                    	for (Expression previousArg: argsList) {
	                                        if (ChainTools.matchRefIdInExpression(exitRefParam.getExpression(), previousArg, proc)) {
	                                            currentExitNode.addPred(targetReturnNode);
	                                            targetReturnNode.addSucc(currentExitNode);
	                                        }
                                    	}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void setGlobalDefInfoToPSG(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        // set local def information at call
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> callList = cfgNode.getData("psg_call_global");
                if (callList == null) {
                    continue;
                }
                for (DFANode callNode : callList) {
                    BitSet inSet = cfgNode.getData("InSet");
                    callNode.putData("DefInSet", (BitSet) inSet.clone());
//                    System.out.println("[" + proc.getSymbolName() + "][gcall]def added: [" + inSet.toString() + "], to call node: " + ((DFANode) node.getData("cfg_node")).getData("ir"));
                }
            }
        }

        // set local def information at exit cfgNode
        for (Procedure proc : globalProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
//                System.out.println("cfgNode: " + cfgNode);
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    // exit cfgNode
                    DFANode exitNode = cfgNode.getData("psg_exit_global");
//                    System.out.println("exitNode: " + exitNode + ", proc: " + proc.getSymbolName());
                    BitSet inSet = cfgNode.getData("InSet");
                    exitNode.putData("DefInSet", (BitSet) inSet.clone());
//                    System.out.println("[" + proc.getSymbolName() + "][gexit]def added: [" + inSet.toString() + "], to call node: " + ((DFANode) exitNode.getData("cfg_node")).getData("ir"));
                }
            }
        }
    }

    private void setGlobalUseInfoToPSG(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        // set local use information at return
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> returnList = cfgNode.getData("psg_return_global");
                if (returnList == null) {
                    continue;
                }
                for (DFANode returnNode : returnList) {
                    BitSet outSet = cfgNode.getData("OutSet");
                    returnNode.putData("UseOutSet", (BitSet) outSet.clone());
//                    System.out.println("[" + proc.getSymbolName() + "][greturn]use added: [" + outSet.toString() + "], to return node: " + ((DFANode) returnNode.getData("cfg_node")).getData("ir"));
                }
            }
        }

        // set local use information at entry
        for (Procedure proc : globalProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            DFANode cfgNode = cfg.getEntry();
//            Iterator cfgIter = cfg.iterator();
//            while (cfgIter.hasNext()) {
//                DFANode cfgNode = (DFANode) cfgIter.next();
//                if (ChainTools.isEntryNode(cfgNode)) {
            // entry cfgNode
            DFANode entryNode = cfgNode.getData("psg_entry_global");
            BitSet outSet = cfgNode.getData("OutSet");
            entryNode.putData("UseOutSet", (BitSet) outSet.clone());
//                    System.out.println("[" + proc.getSymbolName() + "][gentry]use added: [" + outSet.toString() + "], to entry node: " + ((DFANode) entryNode.getData("cfg_node")).getData("ir"));
//                }
//            }
        }
    }

    private void setGlobalKillSetInfoToPSG(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        // set local def information at call
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> callList = cfgNode.getData("psg_call_global");
                if (callList == null) {
                    continue;
                }
                for (DFANode callNode : callList) {
                    BitSet inSet = cfgNode.getData("InSet");
                    callNode.putData("DefKillSet", (BitSet) inSet.clone());
                    // Compute UseKillSet
                    BitSet useKillBitSet = new BitSet(globalUseList.size());
                    AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseList.size()];
                    globalUseList.toArray(globalUseArray);
                    for (int i = 0; i < inSet.size(); i++) {
                        if (inSet.get(i)) {
                            for (int j = 0; j < globalUseArray.length; j++) {
                                if (ChainTools.matchIdInExpression(targetArray[i].getExpression(), globalUseArray[j].getExpression())) {
                                    useKillBitSet.set(j);
                                }
                            }
                        }
                    }
                    // store must kill info for Use into predecessor node
                    // this means in this region, among the OUTuse, UseKillSet will be killed
                    Set<DFANode> predSet = callNode.getPreds();
                    for (DFANode pred : predSet) {
                        pred.putData("UseKillSet", useKillBitSet);
                    }
//                    System.out.println("[" + proc.getSymbolName() + "][gcall]kill added: [" + inSet.toString() + "], to call node: " + ((DFANode) node.getData("cfg_node")).getData("ir"));
                }
            }
        }

        // set local def information at exit cfgNode
        for (Procedure proc : globalProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    // exit cfgNode
                    DFANode exitNode = cfgNode.getData("psg_exit_global");
                    BitSet inSet = cfgNode.getData("InSet");
                    exitNode.putData("DefKillSet", (BitSet) inSet.clone());
                    // Compute UseKillSet
                    BitSet useKillBitSet = new BitSet(globalUseList.size());
                    AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseList.size()];
                    globalUseList.toArray(globalUseArray);
                    for (int i = 0; i < inSet.size(); i++) {
                        if (inSet.get(i)) {
                            for (int j = 0; j < globalUseArray.length; j++) {
                                if (ChainTools.matchIdInExpression(targetArray[i].getExpression(), globalUseArray[j].getExpression())) {
                                    useKillBitSet.set(j);
                                }
                            }
                        }
                    }
                    // store must kill info for Use into predecessor node
                    // this means in this region, among the OUTuse, UseKillSet will be killed
                    Set<DFANode> predSet = exitNode.getPreds();
                    for (DFANode pred : predSet) {
                        pred.putData("UseKillSet", useKillBitSet);
                    }
//                    System.out.println("[" + proc.getSymbolName() + "][gexit]kill added: [" + inSet.toString() + "], to call node: " + ((DFANode) exitNode.getData("cfg_node")).getData("ir"));
                }
            }
        }
    }

    private void printPSGForRef() {
        for (Procedure proc : procList) {
            System.out.println("### Ref PSG for " + proc.getSymbolName());
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                System.out.println("## IR: " + CFGraph.getIR(cfgNode));
                // call node
                Set<DFANode> callList = cfgNode.getData("psg_call_ref");
                if (callList != null) {
                    for (DFANode currentCallNode : callList) {
                        System.out.print("Call Node  : ");
                        printNodeIteratively(currentCallNode);
                        System.out.println("");
                    }
                } else {
                    System.out.println("Call Node  : no psg_call_ref");
                }
                // return node
                Set<DFANode> returnList = cfgNode.getData("psg_return_ref");
                if (returnList != null) {
                    for (DFANode currentReturnNode : returnList) {
                        System.out.print("Return Node: ");
                        printNodeIteratively(currentReturnNode);
                        System.out.println("");
                    }
                } else {
                    System.out.println("Return Node: no psg_return_ref");
                }
                // entry node
                Set<DFANode> entryList = cfgNode.getData("psg_entry_ref");
                if (entryList != null) {
                    for (DFANode currentEntryNode : entryList) {
                        System.out.print("Entry Node : ");
                        printNodeIteratively(currentEntryNode);
                        System.out.println("");
                    }
                } else {
                    System.out.println("Entry Node : no psg_entry_ref");
                }
                // exit node
                Set<DFANode> exitList = cfgNode.getData("psg_exit_ref");
                if (exitList != null) {
                    for (DFANode currentExitNode : exitList) {
                        System.out.print("Exit Node  : ");
                        printNodeIteratively(currentExitNode);
                        System.out.println("");
                    }
                } else {
                    System.out.println("Exit Node  : no psg_exit_ref");
                }
            }
        }
    }

    private void printNodeIteratively(DFANode node) {
        if (node.getData("psg_type").equals("entry") || node.getData("psg_type").equals("exit")) {
            System.out.print("printNodeIteratively[" + ((AnalysisTarget) node.getData("param")).getExpression() + ", " + CFGraph.getIR((DFANode) node.getData("cfg_node")) + "]");
        }
        LinkedList workList = new LinkedList();
        HashSet processedList = new HashSet();
        workList.add(node);
        while (workList.isEmpty() == false) {
            DFANode currentNode = (DFANode) workList.removeLast();
            System.out.print(" --> " + CFGraph.getIR((DFANode) currentNode.getData("cfg_node")) + "[" + currentNode.getData("psg_type") + "]");
            if (currentNode.getSuccs() != null) {
                Set<DFANode> succs = currentNode.getSuccs();
                for (DFANode succ : succs) {
                    if (processedList.contains(succ) == false) {
                        workList.add(succ);
                        processedList.add(currentNode);
                    }
                }
            }
        }
        System.out.println("");
    }

    private void printPSGForGlobal() {
        for (Procedure proc : procList) {
            System.out.println("### Global PSG for " + proc.getSymbolName());
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                System.out.println("## IR: " + CFGraph.getIR(cfgNode));
                // call node
                Set<DFANode> callList = cfgNode.getData("psg_call_global");
                if (callList != null) {
                    for (DFANode currentCallNode : callList) {
                        System.out.print("Call Node  : ");
//                        printNodeRecursively(currentCallNode);
                        System.out.println("");
                    }
                } else {
                    System.out.println("Call Node  : no psg_call_global");
                }
                // return node
                Set<DFANode> returnList = cfgNode.getData("psg_return_global");
                if (returnList != null) {
                    for (DFANode currentReturnNode : returnList) {
                        System.out.print("Return Node: ");
//                        printNodeRecursively(currentReturnNode);
                        System.out.println("");
                    }
                } else {
                    System.out.println("Return Node: no psg_return_global");
                }
                // entry node
                DFANode entryNode = cfgNode.getData("psg_entry_global");
                if (entryNode != null) {
                    System.out.print("Entry Node : ");
//                    printNodeRecursively(entryNode);
                    System.out.println("");
                } else {
                    System.out.println("Entry Node : no psg_entry_global");
                }
                // exit node
                DFANode exitNode = cfgNode.getData("psg_exit_global");
                if (exitNode != null) {
                    System.out.print("Exit Node  : ");
//                    printNodeRecursively(exitNode);
                    System.out.println("");
                } else {
                    System.out.println("Exit Node  : no psg_exit_global");
                }
            }
        }
    }

    private void addReachingEdgeForGlobal(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        // add reaching edges at call node
        for (Procedure proc : globalProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
                if (fcList == null || fcList.size() == 0) {
                    continue;
                }
                Set<DFANode> callList = cfgNode.getData("psg_call_global");
                if (callList == null) {
                    continue;
                }
                for (DFANode currentCallNode : callList) {
                    BitSet inSet = cfgNode.getData("InSet");
                    boolean checkEntry = true;
                    for (int idx = 0; idx < targetArray.length; idx++) {
                        AnalysisTarget target = targetArray[idx];
                        if (inSet.get(idx)) {
                            // return node --> call node
                            Set<DFANode> targetReturnSet = target.getDFANode().getData("psg_return_global");
                            for (DFANode targetReturnNode : targetReturnSet) {
                                currentCallNode.addPred(targetReturnNode);
                                targetReturnNode.addSucc(currentCallNode);
                                checkEntry = false;
                            }
                        }
                    }
                    if (checkEntry) {
                        CFGraph cfgTmp = cfgMap.get(proc);
                        DFANode entryNode = cfgTmp.getEntry();
                        DFANode targetEntryNode = entryNode.getData("psg_entry_global");
                        if (targetEntryNode != null) {
                            currentCallNode.addPred(targetEntryNode);
                            targetEntryNode.addSucc(currentCallNode);
                        }
                    }
                }
            }
        }
        // add reaching edges at exit node
        for (Procedure proc : globalProcSet) {
            CFGraph cfg = cfgMap.get(proc);
            Set<AnalysisTarget> targetList = targetListMap.get(proc);
            AnalysisTarget targetArray[] = new AnalysisTarget[targetList.size()];
            targetList.toArray(targetArray);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                if (currentIR == null) {
                    continue;
                }
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    DFANode exitNode = cfgNode.getData("psg_exit_global");
                    // exit cfgNode
                    BitSet inSet = cfgNode.getData("InSet");
                    boolean checkEntry = true;
                    for (int idx = 0; idx < targetArray.length; idx++) {
                        AnalysisTarget target = targetArray[idx];
                        if (inSet.get(idx)) {
                            // return node --> exit node
                            Set<DFANode> targetReturnSet = target.getDFANode().getData("psg_return_global");
                            for (DFANode targetReturnNode : targetReturnSet) {
                                exitNode.addPred(targetReturnNode);
                                targetReturnNode.addSucc(exitNode);
                                checkEntry = false;
                            }
                        }
                    }
                    if (checkEntry) {
                        CFGraph cfgTmp = cfgMap.get(proc);
                        DFANode entryNode = cfgTmp.getEntry();
                        DFANode targetEntryNode = entryNode.getData("psg_entry_global");
                        if (targetEntryNode != null) {
                            exitNode.addPred(targetEntryNode);
                            targetEntryNode.addSucc(exitNode);
                        }
                    }
                }
            }
        }
    }

    private void propagateDefUseData() {
        propagatePhaseOne();
        propagaetPhaseTwo();
    }

    private void printPSGSummaryDetail() {
        for (Procedure proc : procList) {
            System.out.println("############### PSG Summary Detail[" + proc.getDeclarator() + "] #################");
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                // process entry node
                if (ChainTools.isEntryNode(cfgNode, cfg)) {
                    System.out.println("## Entry Node ##");
                    System.out.println("  # Ref Info (psg_entry_ref, use)");
                    Set<DFANode> entryNodeSetRef = cfgNode.getData("psg_entry_ref");
                    if (entryNodeSetRef != null) {
                        for (DFANode entryNode : entryNodeSetRef) {
//                            workListUniverseRef.add(entryNode);
                            Set<AnalysisTarget> useSet = entryNode.getData("use");
                            if (useSet != null) {
                                for (AnalysisTarget use : useSet) {
                                    System.out.println("  -USE: " + use.getExpression() + ", NODE: " + use.getDFANode().getData("ir"));
                                }
                            }
                        }
                    } else {
                        System.out.println("  psg_entry_ref is null");
                    }
                    System.out.println("  # Global Info (psg_entry_global, UseOutSet)");
                    DFANode entryNodeGlobal = cfgNode.getData("psg_entry_global");
                    if (entryNodeGlobal != null) {
//                        workListUniverseGlobal.add(entryNodeGlobal);
                        BitSet useBitSet = entryNodeGlobal.getData("UseOutSet");
                        AnalysisTarget[] useSet = new AnalysisTarget[globalUseList.size()];
                        globalUseList.toArray(useSet);
                        for (int idx = 0; idx < useSet.length; idx++) {
                            if (useBitSet.get(idx)) {
                                System.out.println("  -USE: " + useSet[idx].getExpression() + ", NODE: " + useSet[idx].getDFANode().getData("ir"));
                            }
                        }
                    } else {
                        System.out.println("  psg_entry_global is null");
                    }
                }
                if (currentIR == null) {
                    continue;
                }
                // process call & return node
                if (IRTools.containsFunctionCall(currentIR)) {
                    // call node
                    System.out.println("## Call Node ## IR: " + currentIR);
                    System.out.println("  # Ref Info (psg_call_ref, def)");
                    Set<DFANode> callNodeSetRef = cfgNode.getData("psg_call_ref");
                    if (callNodeSetRef != null) {
                        for (DFANode callNode : callNodeSetRef) {
//                            workListUniverseRef.add(node);
                            Set<AnalysisTarget> defList = callNode.getData("def");
                            if (defList != null) {
                                for (AnalysisTarget def : defList) {
                                    System.out.println("  -DEF: " + def.getExpression() + ", NODE: " + def.getDFANode().getData("ir"));
                                }
                            }
                        }
                    } else {
                        System.out.println("  psg_call_ref is null");
                    }
                    System.out.println("  # Global Info (psg_call_global, DefInSet)");
                    Set<DFANode> callNodeSetGlobal = cfgNode.getData("psg_call_global");
                    if (callNodeSetGlobal != null) {
                        for (DFANode callNode : callNodeSetGlobal) {
//                            workListUniverseGlobal.add(node);
                            BitSet defBitSet = callNode.getData("DefInSet");
                            AnalysisTarget[] defSet = new AnalysisTarget[globalDefList.size()];
                            globalDefList.toArray(defSet);
                            for (int idx = 0; idx < defSet.length; idx++) {
                                if (defBitSet.get(idx)) {
                                    System.out.println("  -DEF: " + defSet[idx].getExpression() + ", NODE: " + defSet[idx].getDFANode().getData("ir"));
                                }
                            }
                        }
                    } else {
                        System.out.println("  psg_call_global is null");
                    }
                    // return node
                    System.out.println("## Return Node ## IR: " + currentIR);
                    System.out.println("  # Ref Info (psg_return_ref, use)");
                    Set<DFANode> returnNodeSetRef = cfgNode.getData("psg_return_ref");
                    if (returnNodeSetRef != null) {
                        for (DFANode returnNode : returnNodeSetRef) {
//                            workListUniverseRef.add(returnNode);
                            Set<AnalysisTarget> useList = returnNode.getData("use");
                            if (useList != null) {
                                for (AnalysisTarget use : useList) {
                                    System.out.println("  -USE: " + use.getExpression() + ", NODE: " + use.getDFANode().getData("ir"));
                                }
                            }
                        }
                    } else {
                        System.out.println("  psg_return_ref is null");
                    }
                    System.out.println("  # Global Info (psg_return_global, UseOutSet)");
                    Set<DFANode> returnNodeSetGlobal = cfgNode.getData("psg_return_global");
                    if (returnNodeSetGlobal != null) {
                        for (DFANode returnNode : returnNodeSetGlobal) {
//                            workListUniverseGlobal.add(returnNode);
                            BitSet useBitSet = returnNode.getData("UseOutSet");
                            AnalysisTarget[] useSet = new AnalysisTarget[globalUseList.size()];
                            globalUseList.toArray(useSet);
                            for (int idx = 0; idx < useSet.length; idx++) {
                                if (useBitSet.get(idx)) {
                                    System.out.println("  -USE: " + useSet[idx].getExpression() + ", NODE: " + useSet[idx].getDFANode().getData("ir"));
                                }
                            }
                        }
                    } else {
                        System.out.println("  psg_return_global is null");
                    }
                }
                // process exit node
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    System.out.println("## Exit Node ##");
                    System.out.println("  # Ref Info (psg_exit_ref, def)");
                    Set<DFANode> exitNodeSetRef = cfgNode.getData("psg_exit_ref");
                    if (exitNodeSetRef != null) {
                        for (DFANode exitNode : exitNodeSetRef) {
//                            workListUniverseRef.add(exitNode);
                            Set<AnalysisTarget> defSet = exitNode.getData("def");
                            if (defSet != null) {
                                for (AnalysisTarget def : defSet) {
                                    System.out.println("  -DEF: " + def.getExpression() + ", NODE: " + def.getDFANode().getData("ir"));
                                }
                            }
                        }
                    } else {
                        System.out.println("  psg_exit_ref is null");
                    }
                    System.out.println("  # Global Info (psg_exit_global, DefInSet)");
                    DFANode exitNodeGlobal = cfgNode.getData("psg_exit_global");
                    if (exitNodeGlobal != null) {
//                        workListUniverseGlobal.add(exitNodeGlobal);
                        BitSet defBitSet = exitNodeGlobal.getData("DefInSet");
                        AnalysisTarget[] defSet = new AnalysisTarget[globalDefList.size()];
                        globalDefList.toArray(defSet);
                        for (int idx = 0; idx < defSet.length; idx++) {
                            if (defBitSet.get(idx)) {
                                System.out.println("  -DEF: " + defSet[idx].getExpression() + ", NODE: " + defSet[idx].getDFANode().getData("ir"));
                            }
                        }
                    } else {
                        System.out.println("  psg_exit_global is null");
                    }
                }
            }
        }
    }

    private void collectWorkListUniverseRef() {
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                // process entry node
                if (ChainTools.isEntryNode(cfgNode, cfg)) {
                    Set<DFANode> entryNodeSetRef = cfgNode.getData("psg_entry_ref");
                    if (entryNodeSetRef != null) {
                        for (DFANode entryNode : entryNodeSetRef) {
                            workListUniverseRef.add(entryNode);
                        }
                    }
                }
                if (currentIR == null) {
                    continue;
                }
                // process call & return node
                if (IRTools.containsFunctionCall(currentIR)) {
                    Set<DFANode> callNodeSetRef = cfgNode.getData("psg_call_ref");
                    if (callNodeSetRef != null) {
                        for (DFANode callNode : callNodeSetRef) {
                            workListUniverseRef.add(callNode);
                        }
                    }
                    // return node
                    Set<DFANode> returnNodeSetRef = cfgNode.getData("psg_return_ref");
                    if (returnNodeSetRef != null) {
                        for (DFANode returnNode : returnNodeSetRef) {
                            workListUniverseRef.add(returnNode);
                        }
                    }
                }
                // process exit node
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    Set<DFANode> exitNodeSetRef = cfgNode.getData("psg_exit_ref");
                    if (exitNodeSetRef != null) {
                        for (DFANode exitNode : exitNodeSetRef) {
                            workListUniverseRef.add(exitNode);
                        }
                    }
                }
            }
        }
    }

    private void collectWorkListUniverseGlobal() {
        for (Procedure proc : procList) {
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                // process entry node
                if (ChainTools.isEntryNode(cfgNode, cfg)) {
                    DFANode entryNodeGlobal = cfgNode.getData("psg_entry_global");
                    if (entryNodeGlobal != null) {
                        workListUniverseGlobal.add(entryNodeGlobal);
                    }
                }
                if (currentIR == null) {
                    continue;
                }
                // process call & return node
                if (IRTools.containsFunctionCall(currentIR)) {
                    Set<DFANode> callNodeSetGlobal = cfgNode.getData("psg_call_global");
                    if (callNodeSetGlobal != null) {
                        for (DFANode callNode : callNodeSetGlobal) {
                            workListUniverseGlobal.add(callNode);
                        }
                    }
                    // return node
                    Set<DFANode> returnNodeSetGlobal = cfgNode.getData("psg_return_global");
                    if (returnNodeSetGlobal != null) {
                        for (DFANode returnNode : returnNodeSetGlobal) {
                            workListUniverseGlobal.add(returnNode);
                        }
                    }
                }
                // process exit node
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    DFANode exitNodeGlobal = cfgNode.getData("psg_exit_global");
                    if (exitNodeGlobal != null) {
                        workListUniverseGlobal.add(exitNodeGlobal);
                    }
                }
            }
        }
    }

    private void propagatePhaseOne() {
        // for DEF of Ref
        // NODESET = {call,return,exit}
        // EDGESET = {all edges}
        LinkedList<DFANode> workListRef = new LinkedList();
        workListRef.addAll(workListUniverseRef);
        for (DFANode node : workListUniverseRef) {
            // initialize
            Set<AnalysisTarget> inDefSet = new HashSet();
            node.putData("INdef", inDefSet);
            Set<AnalysisTarget> outDefSet = new HashSet();
            Set<AnalysisTarget> defSet = node.getData("def");
            if (defSet != null) {
                outDefSet.addAll(defSet);
            }
            node.putData("OUTdef", outDefSet);
            // entry node is not considered at PhaseOne
            if (node.getData("psg_type").equals("entry")) {
                workListRef.remove(node);
            }
        }
        while (workListRef.isEmpty() == false) {
            DFANode node = workListRef.removeFirst();
            Set<AnalysisTarget> previousOutDefSet = new HashSet();
            Set<AnalysisTarget> outDefSet = node.getData("OUTdef");
            previousOutDefSet.addAll(outDefSet);
            Set<AnalysisTarget> inDefSet = node.getData("INdef");
            Set<DFANode> predSet = node.getPreds();
            for (DFANode pred : predSet) {
                if (pred.getData("OUTdef") != null) {
                    inDefSet.addAll((Set<AnalysisTarget>) pred.getData("OUTdef"));
                }
            }
            outDefSet.addAll(inDefSet);
            if (previousOutDefSet.equals(outDefSet) == false) {
                Set<DFANode> succSet = node.getSuccs();
                for (DFANode succ : succSet) {
                    if (node.getData("psg_type").equals("entry") == false) {
                        if (workListUniverseRef.contains(succ)) {
                            workListRef.add(succ);
                        }
                    }
                }
            }
        }
        // for USE of Ref
        // NODESET = {entry,call,return}
        // EDGESET = {all edges}
        workListRef = new LinkedList();
        workListRef.addAll(workListUniverseRef);
        for (DFANode node : workListUniverseRef) {
            // initialize
            Set<AnalysisTarget> outUseSet = new HashSet();
            node.putData("OUTuse", outUseSet);
            Set<AnalysisTarget> inUseSet = new HashSet();
            Set<AnalysisTarget> useSet = node.getData("use");
            if (useSet != null) {
                inUseSet.addAll(useSet);
            }
            node.putData("INuse", inUseSet);
            // entry node is not considered at PhaseOne
            if (node.getData("psg_type").equals("exit")) {
                workListRef.remove(node);
            }
        }
        while (workListRef.isEmpty() == false) {
            DFANode node = workListRef.removeFirst();
            Set<AnalysisTarget> previousInUseSet = new HashSet();
            Set<AnalysisTarget> inUseSet = node.getData("INuse");
            if (inUseSet != null) {
                previousInUseSet.addAll(inUseSet);
            }
            Set<AnalysisTarget> outUseSet = node.getData("OUTuse");
            Set<DFANode> succSet = node.getSuccs();
            for (DFANode succ : succSet) {
                if (succ.getData("INuse") != null) {
                    outUseSet.addAll((Set<AnalysisTarget>) succ.getData("INuse"));
                }
            }
            inUseSet.addAll(outUseSet);
            if (previousInUseSet.equals(inUseSet) == false) {
                Set<DFANode> predSet = node.getPreds();
                for (DFANode pred : predSet) {
                    if (node.getData("psg_type").equals("exit") == false) {
                        if (workListUniverseRef.contains(pred)) {
                            workListRef.add(pred);
                        }
                    }
                }
            }
        }
        // for DEF of Global
        // NODESET = {call,return,exit}
        // EDGESET = {all edges}
        LinkedList<DFANode> workListGlobal = new LinkedList();
        workListGlobal.addAll(workListUniverseGlobal);
        for (DFANode node : workListUniverseGlobal) {
            // initialize
            BitSet inDefBitSet = new BitSet(globalDefList.size());
            node.putData("INdef", inDefBitSet);
            BitSet outDefBitSet = new BitSet(globalDefList.size());
            node.putData("OUTdef", outDefBitSet);
            // entry node is not considered at PhaseOne
            if (node.getData("psg_type").equals("entry")) {
                workListGlobal.remove(node);
            }
        }
        while (workListGlobal.isEmpty() == false) {
            DFANode node = workListGlobal.removeFirst();
            BitSet outDefBitSet = node.getData("OUTdef");
            BitSet previousOutDefBitSet = (BitSet) outDefBitSet.clone();
            BitSet inDefBitSet = node.getData("INdef");
            // INdef[n] = INdef[n] or OUTdef[p]
            // OUTdef[n] = INdef[n] or DEF[n] - KILL[n]
            Set<DFANode> predSet = node.getPreds();
            for (DFANode pred : predSet) {
                if (pred.getData("OUTdef") != null) {
                    inDefBitSet.or((BitSet) pred.getData("OUTdef"));
                }
            }
            outDefBitSet.or(inDefBitSet);
            if (node.getData("DefInSet") != null) {
                outDefBitSet.or((BitSet) node.getData("DefInSet"));
            }
            if (node.getData("DefKillSet") != null) {
                outDefBitSet.andNot((BitSet) node.getData("DefKillSet"));
            }
            if (previousOutDefBitSet.equals(outDefBitSet) == false) {
                Set<DFANode> succSet = node.getSuccs();
                for (DFANode succ : succSet) {
                    if (node.getData("psg_type").equals("entry") == false) {
                        if (workListUniverseGlobal.contains(succ)) {
                            workListGlobal.add(succ);
                        }
                    }
                }
            }
        }
        // for USE of Global
        // NODESET = {entry,call,return}
        // EDGESET = {all edges}
        workListGlobal = new LinkedList();
        workListGlobal.addAll(workListUniverseGlobal);
        for (DFANode node : workListUniverseGlobal) {
            // initialize
            BitSet inUseBitSet = new BitSet(globalUseList.size());
            node.putData("INuse", inUseBitSet);
            BitSet outUseBitSet = new BitSet(globalUseList.size());
            node.putData("OUTuse", outUseBitSet);
            // entry node is not considered at PhaseOne
            if (node.getData("psg_type").equals("exit")) {
                workListGlobal.remove(node);
            }
        }
        while (workListGlobal.isEmpty() == false) {
            DFANode node = workListGlobal.removeFirst();
            BitSet inUseBitSet = node.getData("INuse");
            BitSet outUseBitSet = node.getData("OUTuse");
            BitSet previousInUseBitSet = (BitSet) inUseBitSet.clone();
            // OUTuse[n] = OUTuse[n] or INuse[s]
            // INuse[n] = UPEXP[n] or (OUTuse[n] - KILL[n])
            Set<DFANode> succSet = node.getSuccs();
            for (DFANode succ : succSet) {
                if (succ.getData("INuse") != null) {
                    outUseBitSet.or((BitSet) succ.getData("INuse"));
                }
            }
            inUseBitSet.or(outUseBitSet);
            if (node.getData("UseKillSet") != null) {
                inUseBitSet.andNot((BitSet) node.getData("UseKillSet"));
            }
            if (node.getData("UseOutSet") != null) {
                inUseBitSet.or((BitSet) node.getData("UseOutSet"));
            }
            if (previousInUseBitSet.equals(inUseBitSet) == false) {
                Set<DFANode> predSet = node.getPreds();
                for (DFANode pred : predSet) {
                    if (node.getData("psg_type").equals("exit") == false) {
                        if (workListUniverseGlobal.contains(pred)) {
                            workListGlobal.add(pred);
                        }
                    }
                }
            }
        }
    }

    private void propagaetPhaseTwo() {
        // for DEF of Ref
        // NODESET = {all nodes}
        // EDGESET = {call binding, reaching, interreaching}
        LinkedList<DFANode> workListRef = new LinkedList();
        workListRef.addAll(workListUniverseRef);
        while (workListRef.isEmpty() == false) {
            DFANode node = workListRef.removeFirst();
            Set<AnalysisTarget> previousOutDefSet = new HashSet();
            Set<AnalysisTarget> outDefSet = node.getData("OUTdef");
            previousOutDefSet.addAll(outDefSet);
            Set<AnalysisTarget> inDefSet = node.getData("INdef");
            Set<DFANode> predSet = node.getPreds();
            for (DFANode pred : predSet) {
                if (pred.getData("psg_type").equals("exit") == false && pred.getData("OUTdef") != null) {
                    inDefSet.addAll((Set<AnalysisTarget>) pred.getData("OUTdef"));
                }
            }
            outDefSet.addAll(inDefSet);
            if (previousOutDefSet.equals(outDefSet) == false) {
                Set<DFANode> succSet = node.getSuccs();
                for (DFANode succ : succSet) {
                    if (workListUniverseRef.contains(succ)) {
                        workListRef.add(succ);
                    }
                }
            }
        }
        // for USE of Ref
        // NODESET = {all nodes}
        // EDGESET = {return binding, reaching, interreaching}
        workListRef = new LinkedList();
        workListRef.addAll(workListUniverseRef);
        while (workListRef.isEmpty() == false) {
            DFANode node = workListRef.removeFirst();
            Set<AnalysisTarget> previousInUseSet = new HashSet();
            Set<AnalysisTarget> inUseSet = node.getData("INuse");
            previousInUseSet.addAll(inUseSet);
            Set<AnalysisTarget> outUseSet = node.getData("OUTuse");
            Set<DFANode> succSet = node.getSuccs();
            for (DFANode succ : succSet) {
                if (succ.getData("psg_type").equals("entry") == false && succ.getData("INuse") != null) {
                    outUseSet.addAll((Set<AnalysisTarget>) succ.getData("INuse"));
                }
            }
            inUseSet.addAll(outUseSet);
            if (previousInUseSet.equals(inUseSet) == false) {
                Set<DFANode> predSet = node.getPreds();
                for (DFANode pred : predSet) {
                    if (workListUniverseRef.contains(pred)) {
                        workListRef.add(pred);
                    }
                }
            }
        }
        // for DEF of Global
        // NODESET = {all nodes}
        // EDGESET = {call binding, reaching, interreaching}
        LinkedList<DFANode> workListGlobal = new LinkedList();
        workListGlobal.addAll(workListUniverseGlobal);
        while (workListGlobal.isEmpty() == false) {
            DFANode node = workListGlobal.removeFirst();
            BitSet outDefBitSet = node.getData("OUTdef");
            BitSet previousOutDefBitSet = (BitSet) outDefBitSet.clone();
            BitSet inDefBitSet = node.getData("INdef");
            // INdef[n] = INdef[n] or OUTdef[p]
            // OUTdef[n] = INdef[n] or DEF[n] - KILL[n]
            Set<DFANode> predSet = node.getPreds();
            for (DFANode pred : predSet) {
                if (pred.getData("psg_type").equals("exit") == false && pred.getData("OUTdef") != null) {
                    inDefBitSet.or((BitSet) pred.getData("OUTdef"));
                }
            }
            outDefBitSet.or(inDefBitSet);
            if (node.getData("DefInSet") != null) {
                outDefBitSet.or((BitSet) node.getData("DefInSet"));
            }
            if (node.getData("DefKillSet") != null) {
                outDefBitSet.andNot((BitSet) node.getData("DefKillSet"));
            }
            if (previousOutDefBitSet.equals(outDefBitSet) == false) {
                Set<DFANode> succSet = node.getSuccs();
                for (DFANode succ : succSet) {
                    if (workListUniverseGlobal.contains(succ)) {
                        workListGlobal.add(succ);
                    }
                }
            }
        }
        // for USE of Global
        // NODESET = {all nodes}
        // EDGESET = {return binding, reaching, interreaching}
        workListGlobal = new LinkedList();
        workListGlobal.addAll(workListUniverseGlobal);
        while (workListGlobal.isEmpty() == false) {
            DFANode node = workListGlobal.removeFirst();
            BitSet inUseBitSet = node.getData("INuse");
            BitSet outUseBitSet = node.getData("OUTuse");
            BitSet previousInUseBitSet = (BitSet) inUseBitSet.clone();
            // OUTuse[n] = OUTuse[n] or INuse[s]
            // INuse[n] = UPEXP[n] or (OUTuse[n] - KILL[n])
            Set<DFANode> succSet = node.getSuccs();
            for (DFANode succ : succSet) {
                if (succ.getData("psg_type").equals("entry") == false && succ.getData("INuse") != null) {
                    outUseBitSet.or((BitSet) succ.getData("INuse"));
                }
            }
            inUseBitSet.or(outUseBitSet);
            if (node.getData("UseKillSet") != null) {
                inUseBitSet.andNot((BitSet) node.getData("UseKillSet"));
            }
            if (node.getData("UseOutSet") != null) {
                inUseBitSet.or((BitSet) node.getData("UseOutSet"));
            }
            if (previousInUseBitSet.equals(inUseBitSet) == false) {
                Set<DFANode> predSet = node.getPreds();
                for (DFANode pred : predSet) {
                    if (workListUniverseGlobal.contains(pred)) {
                        workListGlobal.add(pred);
                    }
                }
            }
        }
    }

    private void printPSGPropagatedDetail() {
        for (Procedure proc : procList) {
            System.out.println("############### PSG Propagated Detail[" + proc.getDeclarator() + "] #################");
            CFGraph cfg = cfgMap.get(proc);
            Iterator cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
                DFANode cfgNode = (DFANode) cfgIter.next();
                Traversable currentIR = (Traversable) CFGraph.getIR(cfgNode);
                // process entry node
                if (ChainTools.isEntryNode(cfgNode, cfg)) {
                    System.out.println("## Entry Node ##");
                    System.out.println("  # Ref Info (psg_entry_ref)");
                    Set<DFANode> entryNodeSetRef = cfgNode.getData("psg_entry_ref");
                    if (entryNodeSetRef != null) {
                        for (DFANode entryNode : entryNodeSetRef) {
                            System.out.println("  # parameter: " + ((AnalysisTarget) entryNode.getData("param")).getExpression());
                            Set<AnalysisTarget> targetSet = entryNode.getData("INdef");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -INdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = entryNode.getData("OUTdef");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -OUTdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = entryNode.getData("INuse");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -INuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = entryNode.getData("OUTuse");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -OUTuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                        }
                    } else {
                        System.out.println("  psg_entry_ref is null");
                    }
                    System.out.println("  # Global Info (psg_entry_global)");
                    DFANode entryNodeGlobal = cfgNode.getData("psg_entry_global");
                    if (entryNodeGlobal != null) {
                        System.out.println("  -INdef: " + entryNodeGlobal.getData("INdef"));
                        System.out.println("  -OUTdef: " + entryNodeGlobal.getData("OUTdef"));
                        System.out.println("  -INuse: " + entryNodeGlobal.getData("INuse"));
                        System.out.println("  -OUTuse: " + entryNodeGlobal.getData("OUTuse"));
                    } else {
                        System.out.println("  psg_entry_global is null");
                    }
                }
                if (currentIR == null) {
                    continue;
                }
                // process call & return node
                if (IRTools.containsFunctionCall(currentIR)) {
                    // call node
                    System.out.println("## Call Node ## IR: " + currentIR);
                    System.out.println("  # Ref Info (psg_call_ref)");
                    Set<DFANode> callNodeSetRef = cfgNode.getData("psg_call_ref");
                    if (callNodeSetRef != null) {
                        for (DFANode callNode : callNodeSetRef) {
                            Set<AnalysisTarget> targetSet = callNode.getData("INdef");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -INdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = callNode.getData("OUTdef");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -OUTdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = callNode.getData("INuse");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -INuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = callNode.getData("OUTuse");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -OUTuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                        }
                    } else {
                        System.out.println("  psg_call_ref is null");
                    }
                    System.out.println("  # Global Info (psg_call_global)");
                    Set<DFANode> callNodeSetGlobal = cfgNode.getData("psg_call_global");
                    if (callNodeSetGlobal != null) {
                        for (DFANode callNode : callNodeSetGlobal) {
                            System.out.println("  -INdef: " + callNode.getData("INdef"));
                            System.out.println("  -OUTdef: " + callNode.getData("OUTdef"));
                            System.out.println("  -INuse: " + callNode.getData("INuse"));
                            System.out.println("  -OUTuse: " + callNode.getData("OUTuse"));
                        }
                    } else {
                        System.out.println("  psg_call_global is null");
                    }
                    // return node
                    System.out.println("## Return Node ## IR: " + currentIR);
                    System.out.println("  # Ref Info (psg_return_ref)");
                    Set<DFANode> returnNodeSetRef = cfgNode.getData("psg_return_ref");
                    if (returnNodeSetRef != null) {
                        for (DFANode returnNode : returnNodeSetRef) {
                            Set<AnalysisTarget> targetSet = returnNode.getData("INdef");
                            if (targetSet != null) {
                                for (AnalysisTarget target : targetSet) {
                                    System.out.println("  -INdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                                }
                            }
                            targetSet = returnNode.getData("OUTdef");
                            if (targetSet != null) {
                                for (AnalysisTarget target : targetSet) {
                                    System.out.println("  -OUTdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                                }
                            }
                            targetSet = returnNode.getData("INuse");
                            if (targetSet != null) {
                                for (AnalysisTarget target : targetSet) {
                                    System.out.println("  -INuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                                }
                            }
                            targetSet = returnNode.getData("OUTuse");
                            if (targetSet != null) {
                                for (AnalysisTarget target : targetSet) {
                                    System.out.println("  -OUTuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                                }
                            }
                        }
                    } else {
                        System.out.println("  psg_return_ref is null");
                    }
                    System.out.println("  # Global Info (psg_return_global)");
                    Set<DFANode> returnNodeSetGlobal = cfgNode.getData("psg_return_global");
                    if (returnNodeSetGlobal != null) {
                        for (DFANode returnNode : returnNodeSetGlobal) {
                            System.out.println("  -INdef: " + returnNode.getData("INdef"));
                            System.out.println("  -OUTdef: " + returnNode.getData("OUTdef"));
                            System.out.println("  -INuse: " + returnNode.getData("INuse"));
                            System.out.println("  -OUTuse: " + returnNode.getData("OUTuse"));
                        }
                    } else {
                        System.out.println("  psg_return_global is null");
                    }
                }
                // process exit node
                if (ChainTools.isExitNode(cfgNode, cfg)) {
                    System.out.println("## Exit Node ##");
                    System.out.println("  # Ref Info (psg_exit_ref)");
                    Set<DFANode> exitNodeSetRef = cfgNode.getData("psg_exit_ref");
                    if (exitNodeSetRef != null) {
                        for (DFANode exitNode : exitNodeSetRef) {
                            Set<AnalysisTarget> targetSet = exitNode.getData("INdef");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -INdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = exitNode.getData("OUTdef");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -OUTdef: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = exitNode.getData("INuse");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -INuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                            targetSet = exitNode.getData("OUTuse");
                            for (AnalysisTarget target : targetSet) {
                                System.out.println("  -OUTuse: " + target.getExpression() + ", NODE: " + target.getDFANode().getData("ir"));
                            }
                        }
                    } else {
                        System.out.println("  psg_exit_ref is null");
                    }
                    System.out.println("  # Global Info (psg_exit_global)");
                    DFANode exitNodeGlobal = cfgNode.getData("psg_exit_global");
                    if (exitNodeGlobal != null) {
                        System.out.println("  -INdef: " + exitNodeGlobal.getData("INdef"));
                        System.out.println("  -OUTdef: " + exitNodeGlobal.getData("OUTdef"));
                        System.out.println("  -INuse: " + exitNodeGlobal.getData("INuse"));
                        System.out.println("  -OUTuse: " + exitNodeGlobal.getData("OUTuse"));
                    } else {
                        System.out.println("  psg_exit_global is null");
                    }
                }
            }
        }
    }

    public Set<AnalysisTarget> getGlobalDefSet() {
        return globalDefList;
    }

    public Set<AnalysisTarget> getGlobalUseSet() {
        return globalUseList;
    }

    private void addInterReachingEdgeForRef(Map<Procedure, Set<AnalysisTarget>> targetListMap) {
        LinkedList<DFANode> callNodeListRef = new LinkedList();
        for (DFANode node : workListUniverseRef) {
            if (node.getData("psg_type").equals("call")) {
                callNodeListRef.add(node);
            }
        }
        for (DFANode callNode : callNodeListRef) {
            Set<DFANode> succ = callNode.getSuccs();
            if (succ.size() != 1) {
                throw new RuntimeException("call site has more than one children: " + ((DFANode) callNode.getData("cfg_node")).getData("ir"));
            }
            DFANode succNode = succ.iterator().next();
            if (reachReturnNode(callNode, succNode)) {
//                System.out.println("InterReaching Node found!!: " + ((DFANode) callNode.getData("cfg_node")).getData("ir"));
                Set<DFANode> returnNodeListRef = ((DFANode) callNode.getData("cfg_node")).getData("psg_return_ref");
                for (DFANode returnNode : returnNodeListRef) {
                	List<Expression> callNodeArgList = callNode.getData("arg");
                	List<Expression> returnNodeArgList = returnNode.getData("arg");
                	for (Expression callNodeArg: callNodeArgList) {
                		for (Expression returnNodeArg: returnNodeArgList) {
		                    if (ChainTools.isIdentical(callNodeArg, returnNodeArg)) {
		                        callNode.addSucc(returnNode);
		                        returnNode.addPred(succNode);
		                    }
                		}
                	}
                }
            }
        }
    }

    private boolean reachReturnNode(DFANode origin, DFANode child) {
//        if (origin.getData("cfg_node").equals(child.getData("cfg_node"))) {
//            return true;
//        }
//        Set<DFANode> children = child.getSuccs();
//        for (DFANode node: children) {
//            return reachReturnNode(origin, node);
//        }
//        return false;
        LinkedList<DFANode> workList = new LinkedList<DFANode>();
        workList.add(child);
        HashSet<DFANode> visited = new HashSet<DFANode>();
        while (workList.isEmpty() == false) {
            DFANode childNode = workList.removeFirst();
            visited.add(childNode);
            if (origin.getData("cfg_node").equals(childNode.getData("cfg_node"))) {
//                System.out.println("reachReturnNode is true: original: " + origin.getData("cfg_node") + ", child: " + child.getData("cfg_node"));
                return true;
            }
            Set<DFANode> children = childNode.getSuccs();
            if (children != null) {
            	for (DFANode ch: children) {
            		if (visited.contains(ch) == false) {
            			workList.add(ch);
            		}
            	}
            }
        }
        return false;
    }
}

package fcuda.transforms;

import java.util.*;

import fcuda.common.*;
import fcuda.utils.*;
import fcuda.*;

import cetus.hir.*;
import cetus.analysis.*;

/*
 * After task function splitting, we need to clean the kernel function
 * from redundant variable declarations:
 * a) declaration of variables that are defined and used in a single 
 *    task function
 * b) declarations of variables that their definition is ThreadIdx-based
 *
 */

public class CleanKernelDecls extends KernelTransformPass
{

  //Some commonly used names and structures across functions

  public String getPassName()
  {
    return new String("[CleanKernelDecls-FCUDA]");
  }

  public CleanKernelDecls(Program program)
  {
    super(program);
  }


  public void removeDecl(CompoundStatement cmpStmt, IDExpression expr, Procedure proc)
  {

    List<Traversable> statements = cmpStmt.getChildren();
    CompoundStatement procCmp = null;
    if (proc != null)
      procCmp = proc.getBody();

    // Go through block statements
    for (Traversable trv : statements) {
      if (trv instanceof DeclarationStatement) {
        if (Tools.getDescendentsOfType(trv, IDExpression.class).get(0) == expr) {
          Statement stmt = (Statement) trv;
          stmt.detach();
          if (procCmp != null)
            procCmp.addANSIDeclaration(((DeclarationStatement) stmt).getDeclaration().clone());
          break;
        }

      }
    }

  } // removeDecl()


  
  public void moveDecl(CompoundStatement cmpStmt, IDExpression expr)
  {
    List<FunctionCall> fCalls = Tools.getFunctionCalls(cmpStmt);

    String e_str = expr.toString();

    for (FunctionCall fcall : fCalls) {
      System.out.println("fcall:"+fcall.toString());  

      Procedure taskProc = fcall.getProcedure();
      List<Expression> args = fcall.getArguments();
      LinkedList<Expression> newArgs = new LinkedList<Expression>();
      TreeMap<Integer, Declaration> redundantParams = new TreeMap<Integer, Declaration>();
      int idx=0;
      for (Expression arg : args) { 
        newArgs.add(arg.clone());
        if (IRTools.containsExpression(arg, expr)) {

          System.out.println("-arg:"+arg.toString()+" contains "+expr.toString());  

          // *AP* Only handles simple 'IDExpression' arguments
          // *FIXME* Should add support for 'Expression' arguments
          if (arg.toString().equals(e_str)) { // argument is just the IDExpression
            System.out.println("- and are equal");  
            newArgs.removeLast();
            redundantParams.put(idx, taskProc.getParameter(idx));
          }
        }
        idx++;
      }

      if (!redundantParams.isEmpty()) {
        fcall.setArguments(newArgs);
        System.out.println("- declList b4 = "+taskProc.getParameters().toString());  
        for (Integer id : redundantParams.keySet()) {
          Declaration redundDecl = redundantParams.get(id);
          taskProc.removeDeclaration(redundDecl);

          // Remove declaration from kernel procedure
           removeDecl(cmpStmt, expr, taskProc);
        }

        System.out.println("- declList after = "+taskProc.getParameters().toString());  
        break;      
      }
    }
  } // moveDecl()



  public void collectBlockDecls(CompoundStatement cmpStmt, int cur_level, Map<Integer, 
        Map<IDExpression, Integer>> lev2varUseMap, Set<IDExpression> funcCallParams)
  {

    if (funcCallParams == null)
      funcCallParams = new TreeSet<IDExpression>();  

    if (lev2varUseMap == null)
      lev2varUseMap = new HashMap<Integer, Map<IDExpression, Integer>>();


    Map<IDExpression, Integer> var2freqMap = new TreeMap<IDExpression, Integer>();
    lev2varUseMap.put(cur_level, var2freqMap);


    System.out.println("cur_level:"+cur_level);  

    List<Traversable> statements = cmpStmt.getChildren();

    // Go through block statements
    for (Traversable trv : statements) {
      List<Traversable> childs = trv.getChildren();
      if (trv instanceof DeclarationStatement) {
        IDExpression idexpr = (IDExpression)(Tools.getDescendentsOfType(trv, IDExpression.class)).get(0);
        // Do NOT account for implicit variables
        if (idexpr.equals((IDExpression)MCUDAUtils.getBidxID().get(0)) ||
            idexpr.equals((IDExpression)MCUDAUtils.Bdim.getId().get(0)))
          continue;
        var2freqMap.put(idexpr, 0);
      } else if (trv instanceof CompoundStatement) {
        collectBlockDecls((CompoundStatement) trv, cur_level+1, lev2varUseMap, funcCallParams);
      } else if(Tools.containsClass(childs, CompoundStatement.class)) {
        for (Traversable chldTrav : childs) {
          if (chldTrav instanceof CompoundStatement)
            collectBlockDecls((CompoundStatement) chldTrav, cur_level+1, lev2varUseMap, funcCallParams);
          else {
            Set<Expression> refSet = DataFlow.mayDefine(chldTrav);
            refSet.addAll(DataFlow.getUseSet(chldTrav));
            for (Expression expr : refSet) {
              if (expr instanceof ArrayAccess || expr instanceof AccessExpression)
                continue;

              IDExpression idExp = (IDExpression) expr;
              int lev;
              for(lev=cur_level; lev>=0; lev--) {
                Map<IDExpression, Integer> var2freqM = lev2varUseMap.get(lev);            

                if(var2freqM.containsKey(idExp)) {   
                  var2freqM.put(idExp, (var2freqM.get(idExp)).intValue() + 1);
                  break;
                }
              }
            }
          }
        }
      } else {
        List<FunctionCall> fcalls = IRTools.getDescendentsOfType(trv, FunctionCall.class);
        //**AP** Assuming that only a single function call per statement exists.
        //**AP** A new pass for separating function calls in statements may be necessary
        if (fcalls.size() > 1) {
          System.out.println("[FCUDA - Error] Can not handle more than one calls in one statement");
          System.exit(0);
        }

        FunctionCall fcall = null;
        if (fcalls.size() > 0)
          fcall = (FunctionCall) fcalls.get(0); 

        Set<Expression> refSet = DataFlow.mayDefine(trv);
        refSet.addAll(DataFlow.getUseSet(trv));

        System.out.println("Defs+Uses:"+refSet.toString());  

        for (Expression expr : refSet) {

          if (expr instanceof ArrayAccess || expr instanceof AccessExpression ||
              expr instanceof FunctionCall)
            continue;

          IDExpression idExp = (IDExpression) expr;
          int lev;
          for (lev=cur_level; lev>=0; lev--) {
            Map<IDExpression, Integer> var2freqM = lev2varUseMap.get(lev);            

            if (var2freqM.containsKey(idExp)) {   
              var2freqM.put(idExp, (var2freqM.get(idExp)).intValue() + 1);
              break;
            }
          }

          if(fcall != null) {
            if (fcall.getArguments().contains(expr))
              funcCallParams.add(idExp);
          }
        }
      } //else
    } // for (Traversable ...)


    System.out.println("cur_level:"+cur_level);  
    System.out.println("var2freqMap"+var2freqMap.toString());  
    System.out.println("funcCallParams"+funcCallParams.toString());  


    for (IDExpression expr : var2freqMap.keySet()) {
      int ref_num;
      ref_num = var2freqMap.get(expr);
      if (ref_num == 0)
        removeDecl(cmpStmt, expr, null);
      else if (ref_num == 1) {
        if (funcCallParams.contains(expr))
          moveDecl(cmpStmt, expr);
      }
    }

    lev2varUseMap.remove(cur_level);
  } // collectBlockDecls()



  public void transformProcedure(Procedure proc)
  {
    if (!(FCUDAGlobalData.isConstMemProc(proc))) {
      CompoundStatement comp = proc.getBody();
      collectBlockDecls(comp, 0, null, null);
    }
  } // transformProcedure()

}

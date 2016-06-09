//============================================================================//
//    FCUDA
//    Copyright (c) <2016> 
//    <University of Illinois at Urbana-Champaign>
//    <University of California at Los Angeles> 
//    All rights reserved.
// 
//    Developed by:
// 
//        <ES CAD Group & IMPACT Research Group>
//            <University of Illinois at Urbana-Champaign>
//            <http://dchen.ece.illinois.edu/>
//            <http://impact.crhc.illinois.edu/>
// 
//        <VAST Laboratory>
//            <University of California at Los Angeles>
//            <http://vast.cs.ucla.edu/>
// 
//        <Hardware Research Group>
//            <Advanced Digital Sciences Center>
//            <http://adsc.illinois.edu/>
//============================================================================//

package fcuda.transforms;

import java.util.*;

import fcuda.analysis.*;
import fcuda.ir.*;
import fcuda.utils.*;
import fcuda.common.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;
import cetus.application.*;

public class AddHLSPragmas extends KernelTransformPass
{
  // return pass name
  public String getPassName() {
    return new String("[AddHLSPragmas-FCUDA]");
  }

  // constructor
  public AddHLSPragmas(Program program) {
    super(program);
  }

  public void transformProcedure(Procedure proc) {
    // *TAN* Prevent the proc which has constant memory procedure (Stream)
    // to add HLS pragmas. We only need to add HLS pragmas on the constant
    // memory procedure instead.
    // TODO: make a function in FCUDAGlobalData to check whether a procedure
    // has a stream procedure or not.
    Expression constMemKern = FCUDAGlobalData.getConstMemKern();
    if (constMemKern != null) {
      if (proc.getName().toString().equals(constMemKern.toString()))
        return;
    }
    
    List<PragmaAnnotation> pragmaList;
    pragmaList = FCUDAGlobalData.getHLSPragmas(proc);
    if (pragmaList == null)
      return;

    Traversable ch = proc.getBody().getChildren().get(0); 
    if (ch instanceof Annotatable) {
      Annotatable ch1 = (Annotatable)ch;
      int i;
      for (i = 0; i < pragmaList.size();i++) {
        ch1.annotateBefore(pragmaList.get(i));
      }
    } else {
      System.out.println("ERROR exiting...");
    }

  } // transformProcedure()

} // AddHLSPragmas


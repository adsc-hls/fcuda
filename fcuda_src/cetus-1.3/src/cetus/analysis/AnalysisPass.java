package cetus.analysis;

import cetus.hir.IRTools;
import cetus.hir.PrintTools;
import cetus.hir.Program;
import cetus.hir.Tools;

public abstract class AnalysisPass
{
  protected Program program;

  protected AnalysisPass(Program program)
  {
    this.program = program;
  }

  public abstract String getPassName();

  public static void run(AnalysisPass pass)
  {
		double timer = Tools.getTime();
    PrintTools.println(pass.getPassName() + " begin", 0);
    pass.start();
    PrintTools.println(pass.getPassName() + " end in " +
			String.format("%.2f seconds", Tools.getTime(timer)), 0);
    if (!IRTools.checkConsistency(pass.program))
      throw new InternalError("Inconsistent IR after " + pass.getPassName());
  }

  public abstract void start();
}

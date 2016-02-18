package fcuda.transforms;

import cetus.hir.*;
import cetus.exec.*;

import fcuda.utils.*;
import fcuda.ir.*;

import java.util.*;

public class CleanLaunches
{
  private static String pass_name = "[CleanLaunch-MCUDA]";
  private Program program;

  public CleanLaunches(Program program)
  {
    this.program = program;
  }

  public static void run(Program program)
  {
    Tools.printlnStatus(pass_name + " begin", 1);

    CleanLaunches pass = new CleanLaunches(program);
    pass.start();

    Tools.printlnStatus(pass_name + " end", 1);
  }

  private void start()
  {
    DepthFirstIterator iter = new DepthFirstIterator(program);
    iter.pruneOn(VariableDeclaration.class);
    iter.pruneOn(ThreadLoop.class);

    for (;;)
    {
      KernelLaunch call = null;
      try {
        //*cetus-1.1*    call = iter.next(KernelLaunch.class);
        call = (KernelLaunch)iter.next(KernelLaunch.class);
      } catch (NoSuchElementException e) {
        break;
      }

      call.clipLaunchArguments(2);
      if(Driver.getOptionValue("Xpilot") != null) {
        List<Expression> launchArgs = call.getLaunchArguments();
        List<Expression> newLaunchArgs = new LinkedList<Expression>();
        for(int i = 0; i < MCUDAUtils.Gdim.getNumEntries(); i++)
          newLaunchArgs.add(new AccessExpression(launchArgs.get(0), 
                AccessOperator.MEMBER_ACCESS,
                //*cetus-1.1*   new Identifier(MCUDAUtils.Gdim.getDimEntry(i))));
            new NameID(MCUDAUtils.Gdim.getDimEntry(i))));
        for(int i = 0; i < MCUDAUtils.Bdim.getNumEntries(); i++)
          newLaunchArgs.add(new AccessExpression(launchArgs.get(1), 
                AccessOperator.MEMBER_ACCESS,
                //*cetus-1.1*   new Identifier(MCUDAUtils.Gdim.getDimEntry(i))));
            new NameID(MCUDAUtils.Gdim.getDimEntry(i))));
        call.setLaunchArguments(newLaunchArgs);
      } 
    }
  }
}

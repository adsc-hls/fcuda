package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.List;

/**
* Represents an exception handling block (try-catch-finally block) in a C++ or
* Java program.  This class is derived from CompoundStatement because in C++ it
* is legal to have a procedure whose body is an exception handler. An exception
* handler has a try block followed by at least one catch block followed
* optionally by a finally block. This class is not used in C.
*/
public class ExceptionHandler extends CompoundStatement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ExceptionHandler.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    private boolean has_finally;

    /**
    * Creates an exception handler.
    */
    @SuppressWarnings("unchecked")
    public ExceptionHandler(CompoundStatement try_block, List catch_blocks) {
        object_print_method = class_print_method;
        has_finally = false;
        children.add(try_block);
        children.addAll(catch_blocks);
    }

    /**
    * Creates an exception handler.
    */
    @SuppressWarnings("unchecked")
    public ExceptionHandler(CompoundStatement try_block, List catch_blocks,
                            CompoundStatement finally_block) {
        object_print_method = class_print_method;
        has_finally = true;
        children.add(try_block);
        children.addAll(catch_blocks);
        children.add(finally_block);
    }

    /**
    * Appends a catch block to the list of catch blocks.
    *
    * @param catch_block The block to add.
    */
    public void addCatchBlock(CompoundStatement catch_block) {
        int last_catch = children.size();
        if (has_finally) {
            last_catch--;
        }
        children.add(catch_block);
    }

    /**
    * Appends a finally block.
    *
    * @param finally_block The block to add.
    */
    public void addFinallyBlock(CompoundStatement finally_block) {
        if (has_finally) {
            throw new IllegalStateException(
                    "cannot have more than one finally block");
        }
        has_finally = true;
        children.add(finally_block);
    }

    /**
    * Prints a statement to a stream.
    *
    * @param h The statement to print.
    * @param o The writer on which to print the statement.
    */
    public static void defaultPrint(ExceptionHandler h, PrintWriter o) {
        o.println("try");
        int n = h.children.size() - 1;
        if (h.has_finally) {
            n--;
        }
        for (int i = 1; i <= n; i++) {
            o.println("catch");
            h.children.get(i).print(o);
        }
        if (h.has_finally) {
            o.println("finally");
            h.children.get(n + 1).print(o);
        }
    }

}

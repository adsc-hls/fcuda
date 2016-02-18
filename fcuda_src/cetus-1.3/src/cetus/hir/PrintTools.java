package cetus.hir;

import cetus.exec.Driver;

import java.io.PrintWriter;
import java.util.*;

/**
* <b>PrintTools</b> provides tools that perform printing of collections of IR
* or debug messages.
*/
public final class PrintTools {

    // Short names for system properties
    public static final String line_sep = System.getProperty("line.separator");
    public static final String file_sep = System.getProperty("file.separator");
    public static final String path_sep = System.getProperty("path.separator");

    /** Global verbosity taken from the command-line option */
    private static final int verbosity =
        Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();

    private PrintTools() {
    }

    /**
    * Prints a Printable object to System.err if the
    * verbosity level is greater than min_verbosity.
    *
    * @param p A Printable object.
    * @param min_verbosity An integer to compare with the value
    *   set by the -verbosity command-line flag.
    */
    public static void printlnStatus(Printable p, int min_verbosity) {
        if (verbosity >= min_verbosity) {
            System.err.println(p + "");
        }
    }

    /**
    * Prints a string to System.err if the
    * verbosity level is greater than min_verbosity.
    *
    * @param message The message to be printed.
    * @param min_verbosity An integer to compare with the value
    *   set by the -verbosity command-line flag.
    */
    public static void printlnStatus(String message, int min_verbosity) {
        if (verbosity >= min_verbosity) {
            System.err.println(message);
        }
    }

    /**
    * Prints the specified items to {@link System#err} with separating
    * white spaces if verbosity is greater than {@code min_verbosity}.
    * This method minimizes overheads from string composition since it is done
    * only if the verbosity level is met.
    * @param min_verbosity the minium verbosity.
    * @param items the list of items to be printed.
    */
    public static void printlnStatus(int min_verbosity, Object... items) {
        if (min_verbosity <= PrintTools.getVerbosity()) {
            if (items.length > 0) {
                StringBuilder sb = new StringBuilder(80);
                sb.append(items[0]);
                for (int i = 1; i < items.length; i++) {
                    sb.append(" ").append(items[i]);
                }
                System.err.println(sb.toString());
            }
        }
    }

    /**
    * Prints a string to System.out if the
    * verbosity level is greater than min_verbosity.
    *
    * @param message The message to be printed.
    * @param min_verbosity An integer to compare with the value
    *   set by the -verbosity command-line flag.
    */
    public static void print(String message, int min_verbosity) {
        if (verbosity >= min_verbosity) {
            System.out.print(message);
        }
    }

    /**
    * Prints a string to System.out if the
    * verbosity level is greater than min_verbosity.
    *
    * @param message The message to be printed.
    * @param min_verbosity An integer to compare with the value
    *   set by the -verbosity command-line flag.
    */
    public static void println(String message, int min_verbosity) {
        if (verbosity >= min_verbosity) {
            System.out.println(message);
        }
    }

    /**
    * Prints a Printable object to System.err if the
    * verbosity level is greater than min_verbosity.
    *
    * @param p A Printable object.
    * @param min_verbosity An integer to compare with the value
    *   set by the -verbosity command-line flag.
    */
    public static void printStatus(Printable p, int min_verbosity) {
        if (verbosity >= min_verbosity) {
            System.err.print(p.toString());
        }
    }

    /**
    * Prints a string to System.err if the
    * verbosity level is greater than min_verbosity.
    *
    * @param message The message to be printed.
    * @param min_verbosity An integer to compare with the value
    *   set by the -verbosity command-line flag.
    */
    public static void printStatus(String message, int min_verbosity) {
        if (verbosity >= min_verbosity) {
            System.err.print(message);
        }
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating stirng. If the list contains an object not printable, this
    * method throws a cast exception.
    * @param list the list of printable object.
    * @param w the target print writer.
    * @param sep the separating string.
    */
    public static void
            printListWithSeparator(List list, PrintWriter w, String sep) {
        if (list == null) {
            return;
        }
        int list_size = list.size();
        if (list_size > 0) {
            ((Printable)list.get(0)).print(w);
            for (int i = 1; i < list_size; i++) {
                w.print(sep);
                ((Printable)list.get(i)).print(w);
            }
        }
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating comma.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printListWithComma(List list, PrintWriter w) {
        printListWithSeparator(list, w, ", ");
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating white space.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printListWithSpace(List list, PrintWriter w) {
        printListWithSeparator(list, w, " ");
    }

    /**
    * Prints a list of printable object to the specified print writer without
    * any separating string.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printList(List list, PrintWriter w) {
        printListWithSeparator(list, w, "");
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating new line character.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printlnList(List list, PrintWriter w) {
        printListWithSeparator(list, w, line_sep);
        w.println("");
    }

    /** Returns the global verbosity level */
    public static int getVerbosity() {
        return verbosity;
    }

    /**
    * Converts a collection of objects to a string with the given separator.
    * By default, the element of the collections are sorted alphabetically, and
    * any {@code Symbol} object is printed with its name.
    *
    * @param coll the collection to be converted.
    * @param separator the separating string.
    * @return the converted string.
    */
    public static String
            collectionToString(Collection coll, String separator) {
        String ret = "";
        if (coll == null || coll.isEmpty()) {
            return ret;
        }
        // Sort the collection first.
        TreeSet<String> sorted = new TreeSet<String>();
        for (Object o : coll) {
            if (o instanceof Symbol) {
                sorted.add(((Symbol)o).getSymbolName());
            } else {
                sorted.add(o.toString());
            }
        }
        Iterator<String> iter = sorted.iterator();
        if (iter.hasNext()) {
            StringBuilder sb = new StringBuilder(80);
            sb.append(iter.next());
            while (iter.hasNext()) {
                sb.append(separator).append(iter.next());
            }
            ret = sb.toString();
        }
        return ret;
    }

    /**
    * Converts a list of objects to a string with the given separator.
    *
    * @param list the list to be converted.
    * @param separator the separating string.
    * @return the converted string.
    */
    public static String listToString(List list, String separator) {
        if (list == null || list.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(80);
        sb.append(list.get(0));
        int list_size = list.size();
        for (int i = 1; i < list_size; i++) {
            sb.append(separator).append(list.get(i));
        }
        return sb.toString();
    }

    /** 
    * Converts a list of objects to a string. The difference from
    * {@code listToString} is that this method inserts the separating string
    * only if the heading string length is non-zero.
    * @param list the list to be converted.
    * @param separator the separating string.
    */
    public static String listToStringWithSkip(List list, String separator) {
        if (list == null || list.isEmpty()) {
            return "";
        }
        String prev = list.get(0).toString();
        StringBuilder sb = new StringBuilder(80);
        sb.append(prev);
        int list_size = list.size();
        for (int i = 1; i < list_size; i++) {
            if (prev.length() > 0) {
                sb.append(separator);
            }
            prev = list.get(i).toString();
            sb.append(prev);
        }
        return sb.toString();
    }

    /** Converts a map to a string. */
    public static String mapToString(Map map, String separator) {
        if (map == null || map.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(80);
        Iterator iter = map.keySet().iterator();
        Object key = iter.next();
        sb.append(key).append(":").append(map.get(key));
        while (iter.hasNext()) {
            key = iter.next();
            sb.append(separator).append(key).append(":").append(map.get(key));
        }
        return sb.toString();
    }
}

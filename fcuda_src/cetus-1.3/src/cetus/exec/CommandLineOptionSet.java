package cetus.exec;

import java.util.*;
import cetus.hir.PrintTools;

public class CommandLineOptionSet
{
  public final int ANALYSIS = 1;
  public final int TRANSFORM = 2;
  public final int UTILITY = 3;
  public final int CODEGEN = 4;

  private class OptionRecord
  {
    public int option_type;
    public String value;
    //arg seems unnecessary
    public String arg;
    public String usage;
    // extra options for selecting specific hir for inclusion/exclusion.
    private int select_type; // 0 for exclusion 1 for inclusion
    private Map<String, List<String>> select_map;

    public OptionRecord(int type, String usage)
    {
      this.option_type = type;
      this.value = null;
      this.arg = null;
      this.usage = usage;
      select_type = 0;
      select_map = new HashMap<String, List<String>>();
    }

    public OptionRecord(int type, String arg, String usage)
    {
      this.option_type = type;
      this.value = null;
      this.arg = arg;
      this.usage = usage;
      select_type = 0;
      select_map = new HashMap<String, List<String>>();
    }
    public OptionRecord(int type, String value, String arg, String usage)
    {
      this.option_type = type;
      this.value = value;
      this.arg = arg;
      this.usage = usage;
      select_type = 0;
      select_map = new HashMap<String, List<String>>();
    }
  }

  private TreeMap<String, OptionRecord> name_to_record;

  public CommandLineOptionSet()
  {
    name_to_record = new TreeMap<String, OptionRecord>();
  }

  public void add(String name, String usage)
  {
    name_to_record.put(name, new OptionRecord(UTILITY, usage));
  }

  public void add(String name, String arg, String usage)
  {
    name_to_record.put(name, new OptionRecord(UTILITY,
						arg, usage));
  }

  public void add(int type, String name, String usage)
  {
    name_to_record.put(name, new OptionRecord(type, usage));
  }

  public void add(int type, String name, String arg, String usage)
  {
    name_to_record.put(name, new OptionRecord(type, arg, usage));
  }

  public void add(int type, String name, String value, String arg, String usage)
  {
    name_to_record.put(name, new OptionRecord(type, value, arg, usage));
  }

  public boolean contains(String name)
  {
    return name_to_record.containsKey(name);
  }

  public String dumpOptions()
  {
      String retval="";
      for (Map.Entry<String, OptionRecord> stringOptionRecordEntry
              : name_to_record.entrySet()) {
          OptionRecord record = stringOptionRecordEntry.getValue();

          // Print usage
          retval += "#Option: " + stringOptionRecordEntry.getKey();
          retval += "\n#";
          // Print option name and example value
          retval += stringOptionRecordEntry.getKey();
          if (record.arg != null) {
            retval += "=";
            retval += record.arg;
          }
          retval += "\n#";
          retval += record.usage.replaceAll("\n","\n#");
          retval += "\n";

          // Print option name and default value
          retval += stringOptionRecordEntry.getKey();
          if (record.value != null) {
            retval += "=";
            retval += record.value;
          }
          retval += "\n";

      }
      return retval;
  }

    public String getUsage() {
        StringBuilder sb = new StringBuilder(8000);
        String sep = PrintTools.line_sep;
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append("UTILITY").append(sep);
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append(getUsage(UTILITY));
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append("ANALYSIS").append(sep);
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append(getUsage(ANALYSIS));
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append("TRANSFORM").append(sep);
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append(getUsage(TRANSFORM));
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append("CODEGEN").append(sep);
        for (int i = 0; i < 80; i++) sb.append("-");
        sb.append(sep).append(getUsage(CODEGEN));
        return sb.toString();
    }

  public String getUsage(int type)
  {
    String usage = "";

      for (Map.Entry<String, OptionRecord> stringOptionRecordEntry : name_to_record.entrySet()) {
          OptionRecord record = stringOptionRecordEntry.getValue();

          if (record.option_type == type) {
              usage += "-";
              usage += stringOptionRecordEntry.getKey();

              if (record.arg != null) {
                  usage += "=";
                  usage += record.arg;
              }

              usage += "\n    ";
              usage += record.usage;
              usage += "\n\n";
          }
      }

    return usage;
  }

  public String getValue(String name)
  {
    OptionRecord record = name_to_record.get(name);

    if (record == null)
		/*
      return new String();
		*/
			return null;
    else
      return record.value;
  }

  public void setValue(String name, String value)
  {
    OptionRecord record = name_to_record.get(name);

    if (record != null)
      record.value = value;
  }

  public int getType(String name)
  {
    OptionRecord record = name_to_record.get(name);

    if (record == null)
      return 0;
    else
      return record.option_type;
  }

  /**
  * Includes the specified IR type and name in the inclusion set for the
  * specified option name.
  * @param name the affected option name.
  * @param hir_type the IR type.
  * @param hir_name the IR name.
  */
  public void include(String name, String hir_type, String hir_name)
  {
    OptionRecord record = name_to_record.get(name);
    if (record.select_map.get(hir_type)==null)
      record.select_map.put(hir_type, new LinkedList<String>());
    record.select_map.get(hir_type).add(hir_name);
    record.select_type = 1;
  }

  /**
  * Excludes the specified IR type and name in the inclusion set for the
  * specified option name.
  * @param name the affected option name.
  * @param hir_type the IR type.
  * @param hir_name the IR name.
  */
  public void exclude(String name, String hir_type, String hir_name)
  {
    OptionRecord record = name_to_record.get(name);
    if (record.select_map.get(hir_type)==null)
      record.select_map.put(hir_type, new LinkedList<String>());
    record.select_map.get(hir_type).add(hir_name);
    record.select_type = 0;
  }

  public boolean isIncluded(String name, String hir_type, String hir_name)
  {
    OptionRecord record = name_to_record.get(name);
    return (
      (record.select_type == 1 &&
      record.select_map.get(hir_type) != null &&
      record.select_map.get(hir_type).contains(hir_name))
      ||
      (record.select_type == 0 &&
      (record.select_map.get(hir_type) == null ||
      !record.select_map.get(hir_type).contains(hir_name)))
    );
  }

  public boolean isExcluded(String name, String hir_type, String hir_name)
  {
    return !isIncluded(name, hir_type, hir_name);
  }
}

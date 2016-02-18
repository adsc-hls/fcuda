package cetus.exec;
import cetus.hir.TranslationUnit;

public interface CetusParser {
  public TranslationUnit parseFile(String filename,CommandLineOptionSet options);
}

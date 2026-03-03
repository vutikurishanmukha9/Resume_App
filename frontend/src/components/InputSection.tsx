import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, FileText, Shield, Sparkles, Zap, Search, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface InputSectionProps {
  analysisMode: "quick" | "deep";
  onModeChange: (mode: "quick" | "deep") => void;
  onAnalyze: (file: File, jobDescription: string) => void;
  isLoading?: boolean;
  error?: string | null;
}

export const InputSection = ({ analysisMode, onModeChange, onAnalyze, isLoading, error }: InputSectionProps) => {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const maxChars = 5000;

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.type === "application/pdf" || file.type === "text/plain")) {
      setResumeFile(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setResumeFile(file);
  };

  const canAnalyze = resumeFile && jobDescription.trim().length > 20 && !isLoading;

  const handleSubmit = () => {
    if (resumeFile && jobDescription.trim().length > 20) {
      onAnalyze(resumeFile, jobDescription);
    }
  };

  return (
    <section className="py-16 px-6">
      <div className="container mx-auto max-w-4xl">
        {/* Mode toggle */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex justify-center mb-10"
        >
          <div className="inline-flex rounded-xl bg-secondary p-1 border border-border">
            <button
              onClick={() => onModeChange("quick")}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 ${analysisMode === "quick"
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              <Zap className="w-4 h-4" />
              Quick Scan
            </button>
            <button
              onClick={() => onModeChange("deep")}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 ${analysisMode === "deep"
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              <Search className="w-4 h-4" />
              Deep Analysis
            </button>
          </div>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Resume upload */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <label className="text-sm font-medium text-foreground mb-3 block">Resume</label>
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`glass-card p-8 flex flex-col items-center justify-center min-h-[220px] cursor-pointer transition-all duration-300 ${isDragging ? "border-primary bg-primary/5" : "hover:border-primary/40"
                } ${resumeFile ? "border-success/40 bg-success/5" : ""}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt"
                onChange={handleFileSelect}
                className="hidden"
              />
              {resumeFile ? (
                <>
                  <div className="w-12 h-12 rounded-xl bg-success/15 flex items-center justify-center mb-3">
                    <FileText className="w-6 h-6 text-success" />
                  </div>
                  <p className="text-sm font-medium text-foreground">{resumeFile.name}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {(resumeFile.size / 1024).toFixed(1)} KB • Click to change
                  </p>
                </>
              ) : (
                <>
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-3">
                    <Upload className="w-6 h-6 text-primary" />
                  </div>
                  <p className="text-sm font-medium text-foreground">Drop your resume here</p>
                  <p className="text-xs text-muted-foreground mt-1">PDF or TXT • Max 16MB</p>
                </>
              )}
            </div>
          </motion.div>

          {/* Job description */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
          >
            <label className="text-sm font-medium text-foreground mb-3 block">Job Description</label>
            <div className="relative">
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value.slice(0, maxChars))}
                placeholder="Paste the job description you're targeting..."
                className="w-full min-h-[220px] glass-card p-5 text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:border-primary/50 transition-colors"
              />
              <div className="absolute bottom-3 right-4 text-xs text-muted-foreground">
                {jobDescription.length.toLocaleString()}/{maxChars.toLocaleString()}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Error display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 rounded-xl bg-destructive/10 border border-destructive/20 flex items-start gap-3"
          >
            <AlertCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
            <p className="text-sm text-destructive">{error}</p>
          </motion.div>
        )}

        {/* Analyze button */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 flex flex-col items-center gap-4"
        >
          <Button
            variant="hero"
            size="lg"
            disabled={!canAnalyze}
            onClick={handleSubmit}
            className="px-10 py-6 text-base font-semibold rounded-xl"
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 mr-2 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Analyze Resume
              </>
            )}
          </Button>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <Shield className="w-3.5 h-3.5" />
              No data stored
            </span>
            <span className="w-1 h-1 rounded-full bg-border" />
            <span>Analysis takes ~15 seconds</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

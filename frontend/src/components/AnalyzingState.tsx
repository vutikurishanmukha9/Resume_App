import { motion } from "framer-motion";
import { Brain, FileSearch, Sparkles } from "lucide-react";

interface AnalyzingStateProps {
  mode: "quick" | "deep";
}

const steps = [
  { icon: FileSearch, label: "Parsing resume content..." },
  { icon: Brain, label: "Running AI analysis..." },
  { icon: Sparkles, label: "Generating insights..." },
];

export const AnalyzingState = ({ mode }: AnalyzingStateProps) => {
  return (
    <section className="min-h-[80vh] flex items-center justify-center px-6 hero-gradient">
      <div className="text-center">
        {/* Pulsing orb */}
        <motion.div
          className="w-24 h-24 mx-auto mb-10 relative"
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        >
          <div className="absolute inset-0 rounded-full bg-primary/20 blur-xl" />
          <div className="absolute inset-2 rounded-full bg-primary/10 backdrop-blur-sm border border-primary/30 flex items-center justify-center">
            <Brain className="w-8 h-8 text-primary" />
          </div>
        </motion.div>

        <motion.h2
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-2xl font-bold mb-2 text-foreground"
        >
          {mode === "quick" ? "Quick Scanning" : "Deep Analyzing"}...
        </motion.h2>
        <p className="text-muted-foreground mb-10">
          Our AI is reviewing your resume against the job requirements
        </p>

        <div className="flex flex-col items-center gap-4">
          {steps.map((step, i) => (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.8 }}
              className="flex items-center gap-3 text-sm"
            >
              <motion.div
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.8 }}
              >
                <step.icon className="w-4 h-4 text-primary" />
              </motion.div>
              <span className="text-muted-foreground">{step.label}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

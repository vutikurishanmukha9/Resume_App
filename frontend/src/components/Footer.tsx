import { Shield, Cpu, Lock } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="border-t border-border py-10 px-6">
      <div className="container mx-auto max-w-6xl">
        <div className="flex flex-wrap items-center justify-center gap-6 mb-6">
          <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
            <Shield className="w-3.5 h-3.5" />
            No resume data stored
          </span>
          <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
            <Cpu className="w-3.5 h-3.5" />
            Powered by AI
          </span>
          <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
            <Lock className="w-3.5 h-3.5" />
            Encrypted upload
          </span>
        </div>
        <p className="text-center text-xs text-muted-foreground/60"> 2025 ResumeAI. Built to help you land your next role.

        </p>
      </div>
    </footer>);

};
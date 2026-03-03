import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { HeroSection } from "@/components/HeroSection";
import { InputSection } from "@/components/InputSection";
import { AnalyzingState } from "@/components/AnalyzingState";
import { ResultsDashboard } from "@/components/ResultsDashboard";
import { Footer } from "@/components/Footer";
import { analyzeResume } from "@/lib/api";
import type { AnalysisResult } from "@/lib/types";

type AppState = "idle" | "analyzing" | "results";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("idle");
  const [analysisMode, setAnalysisMode] = useState<"quick" | "deep">("deep");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (file: File, jobDescription: string) => {
    setError(null);
    setAppState("analyzing");

    try {
      const analysisResult = await analyzeResume(file, jobDescription, analysisMode);
      console.log("Analysis result:", JSON.stringify(analysisResult, null, 2));

      // Validate result has required fields
      if (!analysisResult || typeof analysisResult.atsScore !== 'number') {
        throw new Error("Invalid response from server - missing ATS score");
      }

      // Ensure all required fields have defaults
      const safeResult: AnalysisResult = {
        atsScore: analysisResult.atsScore || 0,
        predictedTitle: analysisResult.predictedTitle || "Resume Analysis",
        experience: analysisResult.experience || "N/A",
        education: analysisResult.education || "N/A",
        skillMatch: analysisResult.skillMatch || 0,
        recruiterReadability: analysisResult.recruiterReadability || 0,
        industryFit: analysisResult.industryFit || 0,
        matchedKeywords: analysisResult.matchedKeywords || [],
        missingKeywords: {
          critical: analysisResult.missingKeywords?.critical || [],
          important: analysisResult.missingKeywords?.important || [],
          optional: analysisResult.missingKeywords?.optional || [],
        },
        skills: analysisResult.skills || {},
        suggestions: analysisResult.suggestions || [],
        strengthAreas: analysisResult.strengthAreas || [
          { area: "Score", score: analysisResult.atsScore || 0 },
        ],
      };

      setResult(safeResult);
      setAppState("results");
    } catch (err) {
      console.error("Analysis failed:", err);
      const message = err instanceof Error ? err.message : "An unexpected error occurred. Please try again.";

      if (message.includes("503") || message.includes("initializing")) {
        setError("The AI models are still loading. Please wait a moment and try again.");
      } else if (message.includes("429") || message.includes("rate")) {
        setError("Too many requests. Please wait a minute before trying again.");
      } else if (message.includes("fetch") || message.includes("network") || message.includes("Failed to fetch")) {
        setError("Cannot connect to the server. Make sure the backend is running on port 5000.");
      } else {
        setError(message);
      }

      setAppState("idle");
    }
  };

  const handleReanalyze = () => {
    setResult(null);
    setError(null);
    setAppState("idle");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      {appState === "idle" && (
        <>
          <HeroSection />
          <InputSection
            analysisMode={analysisMode}
            onModeChange={setAnalysisMode}
            onAnalyze={handleAnalyze}
            isLoading={false}
            error={error}
          />
        </>
      )}
      {appState === "analyzing" && <AnalyzingState mode={analysisMode} />}
      {appState === "results" && result && (
        <ResultsDashboard result={result} onReanalyze={handleReanalyze} />
      )}
      <Footer />
    </div>
  );
};

export default Index;

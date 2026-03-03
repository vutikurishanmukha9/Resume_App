export interface AnalysisResult {
    atsScore: number;
    predictedTitle: string;
    experience: string;
    education: string;
    skillMatch: number;
    recruiterReadability: number;
    industryFit: number;
    matchedKeywords: string[];
    missingKeywords: {
        critical: string[];
        important: string[];
        optional: string[];
    };
    skills: Record<string, { matched: string[]; missing: string[] }>;
    suggestions: { title: string; description: string; impact: 'high' | 'medium' | 'low' }[];
    strengthAreas: { area: string; score: number }[];
}

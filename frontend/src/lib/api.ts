/**
 * API Service Layer
 * 
 * Connects the React frontend to the FastAPI backend.
 * Transforms backend responses to match the AnalysisResult interface.
 */

import type { AnalysisResult } from './types';

// ─── API Endpoints ───────────────────────────────────────────────
const ENDPOINTS = {
    UPLOAD: '/upload',
    MATCH: '/match_jd_resume',
    ATS_SCORE: '/ats_score',
    ANALYZE_FULL: '/analyze-full',
    HEALTH: '/health',
    READY: '/ready',
};

// ─── Types for backend responses ─────────────────────────────────

interface UploadResponse {
    success: boolean;
    predicted_job: string;
    matches: { title: string; score: string }[];
    salary: string;
    salary_details: {
        confidence: number;
        features: {
            years_experience: number;
            education_level: number;
            seniority_level: number;
            skills_count: number;
        };
        note?: string;
    };
    error?: string;
}

interface JDMatchResponse {
    success: boolean;
    match_percentage: number;
    component_scores: {
        semantic: number;
        keyword: number;
        skills: number;
        context: number;
    };
    missing_keywords: {
        critical?: string[];
        important?: string[];
        optional?: string[];
    };
    keyword_suggestions: string[];
    skills_breakdown: {
        resume_skills: Record<string, string[]>;
        missing_skills: Record<string, string[]>;
        matched_skills: Record<string, string[]>;
    };
    message: string;
    error?: string;
}

interface ATSResponse {
    success: boolean;
    ats_score: number;
    mode: string;
    interpretation: {
        badge: string;
        message: string;
    };
    sub_scores: Record<string, number>;
    matched_keywords: (string | { keyword: string; importance: string })[];
    missing_keywords: {
        critical?: string[];
        important?: string[];
        optional?: string[];
    };
    achievements_found: string[];
    suggestions: string[];
    error?: string;
}

// ─── Education label mapping ─────────────────────────────────────
// Backend returns: 0=Unknown, 1=Bachelor's, 2=Master's, 3=PhD
function getEducationLabel(level: number): string {
    const labels: Record<number, string> = {
        0: 'Not specified',
        1: "Bachelor's Degree",
        2: "Master's Degree",
        3: 'Ph.D.',
    };
    return labels[level] || 'Unknown';
}

// ─── Experience label ────────────────────────────────────────────
function getExperienceLabel(years: number): string {
    if (years === 0) return 'Entry Level';
    if (years < 1) {
        const months = Math.round(years * 12);
        return `${months} month${months !== 1 ? 's' : ''}`;
    }
    if (years < 2) return `${years}+ year${years === 1 ? '' : 's'}`;
    return `${Math.round(years)}+ years`;
}

// ─── Transform backend skills_breakdown to frontend format ───────
function transformSkillsBreakdown(
    breakdown: JDMatchResponse['skills_breakdown'] | undefined
): Record<string, { matched: string[]; missing: string[] }> {
    if (!breakdown) return {};

    const result: Record<string, { matched: string[]; missing: string[] }> = {};

    // Collect all category names from all three sub-objects
    const allCategories = new Set<string>();
    if (breakdown.resume_skills) {
        Object.keys(breakdown.resume_skills).forEach(c => allCategories.add(c));
    }
    if (breakdown.matched_skills) {
        Object.keys(breakdown.matched_skills).forEach(c => allCategories.add(c));
    }
    if (breakdown.missing_skills) {
        Object.keys(breakdown.missing_skills).forEach(c => allCategories.add(c));
    }

    // Build the categorized output
    for (const category of allCategories) {
        const matched = breakdown.matched_skills?.[category] || [];
        const missing = breakdown.missing_skills?.[category] || [];

        // Only include categories that have actual data
        if (matched.length > 0 || missing.length > 0) {
            result[category] = { matched, missing };
        }
    }

    // If no categorized data, fall back to resume_skills as "All Skills"
    if (Object.keys(result).length === 0 && breakdown.resume_skills) {
        const allResumeSkills: string[] = [];
        for (const skills of Object.values(breakdown.resume_skills)) {
            allResumeSkills.push(...(skills || []));
        }
        if (allResumeSkills.length > 0) {
            result['All Skills'] = { matched: allResumeSkills, missing: [] };
        }
    }

    return result;
}

// ─── Transform ATS response to AnalysisResult ────────────────────
function transformATSResponse(atsData: ATSResponse, uploadData?: UploadResponse): AnalysisResult {
    const matchedKeywords = atsData.matched_keywords?.map(kw =>
        typeof kw === 'object' ? kw.keyword : kw
    ) || [];

    const subScores = atsData.sub_scores || {};

    return {
        atsScore: atsData.ats_score,
        predictedTitle: uploadData?.predicted_job || atsData.interpretation?.badge || 'N/A',
        experience: uploadData?.salary_details?.features
            ? getExperienceLabel(uploadData.salary_details.features.years_experience)
            : `${subScores.experience || 0}% match`,
        education: uploadData?.salary_details?.features
            ? getEducationLabel(uploadData.salary_details.features.education_level)
            : `${subScores.education || 0}% match`,
        skillMatch: subScores.skill_match || 0,
        recruiterReadability: 100 - (subScores.formatting_penalty || 0),
        industryFit: subScores.title_match || 0,
        matchedKeywords,
        missingKeywords: {
            critical: atsData.missing_keywords?.critical || [],
            important: atsData.missing_keywords?.important || [],
            optional: atsData.missing_keywords?.optional || [],
        },
        skills: {},
        suggestions: (atsData.suggestions || []).map((s, i) => ({
            title: `Suggestion ${i + 1}`,
            description: s,
            impact: i < 2 ? 'high' as const : i < 4 ? 'medium' as const : 'low' as const,
        })),
        strengthAreas: [
            { area: 'Skill Match', score: subScores.skill_match || 0 },
            { area: 'Title Match', score: subScores.title_match || 0 },
            { area: 'Experience', score: subScores.experience || 0 },
            { area: 'Achievement', score: subScores.achievement || 0 },
            { area: 'Education', score: subScores.education || 0 },
            { area: 'Formatting', score: 100 - (subScores.formatting_penalty || 0) },
        ],
    };
}

// ─── Transform JD Match response to AnalysisResult ───────────────
function transformJDMatchResponse(jdData: JDMatchResponse, uploadData?: UploadResponse): AnalysisResult {
    const componentScores = jdData.component_scores || { semantic: 0, keyword: 0, skills: 0, context: 0 };

    return {
        atsScore: Math.round(jdData.match_percentage),
        predictedTitle: uploadData?.predicted_job || 'Resume Analysis',
        experience: uploadData?.salary_details?.features
            ? getExperienceLabel(uploadData.salary_details.features.years_experience)
            : 'N/A',
        education: uploadData?.salary_details?.features
            ? getEducationLabel(uploadData.salary_details.features.education_level)
            : 'N/A',
        skillMatch: Math.round(componentScores.skills || 0),
        recruiterReadability: Math.round(componentScores.semantic || 0),
        industryFit: Math.round(componentScores.context || 0),
        matchedKeywords: jdData.keyword_suggestions || [],
        missingKeywords: {
            critical: jdData.missing_keywords?.critical || [],
            important: jdData.missing_keywords?.important || [],
            optional: jdData.missing_keywords?.optional || [],
        },
        skills: transformSkillsBreakdown(jdData.skills_breakdown),
        suggestions: [
            ...(jdData.missing_keywords?.critical?.length ? [{
                title: 'Add Critical Keywords',
                description: `Your resume is missing critical keywords: ${jdData.missing_keywords.critical.slice(0, 5).join(', ')}. Add these to improve your match score.`,
                impact: 'high' as const,
            }] : []),
            ...(jdData.missing_keywords?.important?.length ? [{
                title: 'Include Important Terms',
                description: `Consider adding: ${jdData.missing_keywords.important.slice(0, 5).join(', ')} to strengthen your application.`,
                impact: 'medium' as const,
            }] : []),
            ...(componentScores.keyword < 50 ? [{
                title: 'Improve Keyword Coverage',
                description: `Your keyword match is ${componentScores.keyword?.toFixed(1)}%. Mirror language from the job description more closely.`,
                impact: 'high' as const,
            }] : []),
        ],
        strengthAreas: [
            { area: 'Semantic Match', score: Math.round(componentScores.semantic || 0) },
            { area: 'Keyword Match', score: Math.round(componentScores.keyword || 0) },
            { area: 'Skills Match', score: Math.round(componentScores.skills || 0) },
            { area: 'Context Match', score: Math.round(componentScores.context || 0) },
        ],
    };
}

// ─── Public API functions ────────────────────────────────────────

/**
 * Combined analysis: single /analyze-full call
 *
 * Uploads the file ONCE; the backend parses it once and returns
 * upload analysis + ATS score + JD match in a single response.
 */
export async function analyzeResume(
    file: File,
    jobDescription: string,
    mode: 'quick' | 'deep' = 'deep'
): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('resume', file);
    formData.append('jd_text', jobDescription);
    formData.append('mode', mode);

    const res = await fetch(ENDPOINTS.ANALYZE_FULL, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.error || `Server error (${res.status})`);
    }

    const data = await res.json();
    if (!data.success) {
        throw new Error('Analysis failed');
    }

    // Destructure the combined response
    const uploadData: UploadResponse | undefined = data.upload?.success ? data.upload : undefined;
    const atsData: ATSResponse | undefined = data.ats?.success ? data.ats : undefined;
    const jdData: JDMatchResponse | undefined = data.jd_match?.success ? data.jd_match : undefined;

    // Build result from ATS data first (primary score source)
    let result: AnalysisResult;

    if (atsData) {
        result = transformATSResponse(atsData, uploadData);
    } else if (jdData) {
        result = transformJDMatchResponse(jdData, uploadData);
    } else {
        throw new Error('No analysis results were returned');
    }

    // Enrich with JD match data if available
    if (jdData) {
        result.skills = transformSkillsBreakdown(jdData.skills_breakdown);
        result.recruiterReadability = Math.round(jdData.component_scores?.semantic || result.recruiterReadability);

        if (jdData.missing_keywords) {
            result.missingKeywords = {
                critical: jdData.missing_keywords.critical || [],
                important: jdData.missing_keywords.important || [],
                optional: jdData.missing_keywords.optional || [],
            };
        }

        if (jdData.component_scores) {
            result.strengthAreas = [
                { area: 'Semantic', score: Math.round(jdData.component_scores.semantic || 0) },
                { area: 'Keywords', score: Math.round(jdData.component_scores.keyword || 0) },
                { area: 'Skills', score: Math.round(jdData.component_scores.skills || 0) },
                { area: 'Context', score: Math.round(jdData.component_scores.context || 0) },
                { area: 'Achievement', score: atsData?.sub_scores?.achievement || 0 },
                { area: 'Formatting', score: 100 - (atsData?.sub_scores?.formatting_penalty || 0) },
            ];
        }

        // Merge JD suggestions
        const jdSuggestions = transformJDMatchResponse(jdData, uploadData).suggestions;
        const existingTitles = new Set(result.suggestions.map(s => s.title));
        for (const s of jdSuggestions) {
            if (!existingTitles.has(s.title)) {
                result.suggestions.push(s);
            }
        }
    }

    return result;
}

/**
 * Check if the backend is ready
 */
export async function checkBackendReady(): Promise<boolean> {
    try {
        const res = await fetch(ENDPOINTS.READY);
        const data = await res.json();
        return data.models_loaded === true;
    } catch {
        return false;
    }
}

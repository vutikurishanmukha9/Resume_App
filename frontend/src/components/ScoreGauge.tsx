import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface ScoreGaugeProps {
  score: number;
  size?: number;
}

export const ScoreGauge = ({ score, size = 180 }: ScoreGaugeProps) => {
  const [displayScore, setDisplayScore] = useState(0);
  const strokeWidth = 10;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = (displayScore / 100) * circumference;

  const getScoreColor = (s: number) => {
    if (s >= 80) return "var(--success)";
    if (s >= 60) return "var(--warning)";
    return "var(--destructive)";
  };

  const getScoreLabel = (s: number) => {
    if (s >= 85) return "Recruiter Ready";
    if (s >= 70) return "Good Match";
    if (s >= 50) return "Needs Work";
    return "Likely Rejected";
  };

  useEffect(() => {
    let frame: number;
    const duration = 1500;
    const start = performance.now();

    const animate = (now: number) => {
      const elapsed = now - start;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplayScore(Math.round(eased * score));
      if (t < 1) frame = requestAnimationFrame(animate);
    };

    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [score]);

  const colorHsl = getScoreColor(score);
  const isLow = score < 50;

  return (
    <div className={`flex flex-col items-center ${isLow ? "animate-shake" : ""}`}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="hsl(var(--border))"
            strokeWidth={strokeWidth}
            fill="none"
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={`hsl(${colorHsl})`}
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            style={{
              filter: `drop-shadow(0 0 8px hsl(${colorHsl} / 0.4))`,
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-extrabold text-foreground">{displayScore}</span>
          <span className="text-xs text-muted-foreground font-medium mt-0.5">/ 100</span>
        </div>
      </div>
      <p
        className="mt-4 text-sm font-semibold"
        style={{ color: `hsl(${colorHsl})` }}
      >
        {getScoreLabel(score)}
      </p>
    </div>
  );
};

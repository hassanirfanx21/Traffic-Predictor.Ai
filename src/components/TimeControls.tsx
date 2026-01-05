"use client";

interface TimeControlsProps {
  hour: number;
  day: number;
  incidentFlag: boolean;
  limit: number;
  onHourChange: (hour: number) => void;
  onDayChange: (day: number) => void;
  onIncidentChange: (flag: boolean) => void;
  onLimitChange: (limit: number) => void;
  onRefresh: () => void;
  isLoading: boolean;
}

const DAYS = [
  { value: 1, label: "Mon" },
  { value: 2, label: "Tue" },
  { value: 3, label: "Wed" },
  { value: 4, label: "Thu" },
  { value: 5, label: "Fri" },
  { value: 6, label: "Sat" },
  { value: 7, label: "Sun" },
];

export default function TimeControls({
  hour,
  day,
  limit,
  onHourChange,
  onDayChange,
  onLimitChange,
  onRefresh,
  isLoading,
}: TimeControlsProps) {
  const formatHour = (h: number) => {
    if (h === 0) return "12 AM";
    if (h === 12) return "12 PM";
    return h < 12 ? `${h} AM` : `${h - 12} PM`;
  };

  // Get current day (1-7, Mon-Sun)
  const currentJsDay = new Date().getDay();
  const currentDayValue = currentJsDay === 0 ? 7 : currentJsDay;

  return (
    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-[2rem] p-6 md:px-10 md:py-8 shadow-2xl">
      <div className="flex flex-col gap-8">
        {/* Days Row */}
        <div className="flex justify-between items-center overflow-x-auto pt-4 pb-2 md:pb-0 no-scrollbar">
          {DAYS.map((d) => {
            const isActive = d.value === day;
            const isToday = d.value === currentDayValue;
            return (
              <button
                key={d.value}
                onClick={() => onDayChange(d.value)}
                className={`relative flex flex-col items-center justify-center gap-2 transition-all duration-300 group min-w-[60px] h-[80px] rounded-2xl ${
                  isToday
                    ? "bg-gradient-to-b from-pink-500/20 to-red-500/20 border border-pink-500/30 shadow-[0_0_15px_rgba(236,72,153,0.2)]"
                    : isActive
                    ? "bg-white/10 border border-white/20"
                    : "hover:bg-white/5 border border-transparent"
                } ${
                  isActive ? "scale-110 z-10" : "opacity-60 hover:opacity-100"
                }`}
              >
                {isToday && (
                  <span className="absolute -top-2 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-gradient-to-r from-pink-500 to-red-500 text-[9px] font-bold text-white uppercase tracking-wider shadow-lg whitespace-nowrap">
                    Today
                  </span>
                )}

                <span
                  className={`text-sm font-medium tracking-widest uppercase ${
                    isToday
                      ? "text-pink-200"
                      : isActive
                      ? "text-white"
                      : "text-white/60"
                  }`}
                >
                  {d.label}
                </span>

                {/* Dot Indicator */}
                <div
                  className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ${
                    isActive
                      ? "bg-white shadow-[0_0_10px_white]"
                      : isToday
                      ? "bg-pink-500 shadow-[0_0_8px_rgba(236,72,153,0.8)]"
                      : "bg-white/10 group-hover:bg-white/30"
                  }`}
                />
              </button>
            );
          })}
        </div>

        {/* Hour Slider - Sleek Line */}
        <div className="relative pt-6 pb-8">
          {/* Time Display Floating above thumb */}
          <div
            className="absolute -top-2 transform -translate-x-1/2 transition-all duration-75 pointer-events-none z-20"
            style={{ left: `${(hour / 23) * 100}%` }}
          >
            <div className="text-2xl font-light text-white tracking-tight whitespace-nowrap drop-shadow-lg">
              {formatHour(hour)}
            </div>
          </div>

          {/* Track */}
          <div className="h-[1px] w-full bg-white/20 relative flex items-center">
            {/* Progress */}
            <div
              className="absolute left-0 h-[1px] bg-gradient-to-r from-transparent to-white"
              style={{ width: `${(hour / 23) * 100}%` }}
            />

            {/* Ticks & Labels */}
            {[0, 6, 12, 18, 23].map((tick) => (
              <div
                key={tick}
                className="absolute"
                style={{ left: `${(tick / 23) * 100}%` }}
              >
                <div className="absolute -top-1 w-[1px] h-2 bg-white/20" />
                <div className="absolute top-4 -translate-x-1/2 text-[10px] text-white/30 font-mono whitespace-nowrap">
                  {tick === 0
                    ? "12AM"
                    : tick === 12
                    ? "12PM"
                    : tick > 12
                    ? `${tick - 12}PM`
                    : `${tick}AM`}
                </div>
              </div>
            ))}

            <input
              type="range"
              min="0"
              max="23"
              value={hour}
              onChange={(e) => onHourChange(parseInt(e.target.value))}
              className="absolute inset-0 w-full h-8 opacity-0 cursor-pointer z-10"
            />

            {/* Custom Thumb (Visual only, follows input) */}
            <div
              className="absolute w-4 h-4 bg-white rounded-full shadow-[0_0_15px_rgba(255,255,255,0.5)] transform -translate-x-1/2 pointer-events-none transition-all duration-75"
              style={{ left: `${(hour / 23) * 100}%` }}
            >
              <div className="absolute inset-0 bg-white rounded-full animate-ping opacity-20" />
            </div>
          </div>
        </div>

        {/* Controls Row: Limit, Incident, Predict */}
        <div className="flex flex-col gap-4 pt-4 border-t border-white/5">
          {/* Limit Slider */}
          <div className="flex items-center gap-4">
            <span className="text-xs text-white/40 uppercase tracking-wider w-16">
              Limit
            </span>
            <input
              type="range"
              min="1"
              max="320"
              value={limit}
              onChange={(e) => onLimitChange(parseInt(e.target.value))}
              className="flex-1 h-1 bg-white/10 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
            />
            <span className="text-sm font-mono text-white/60 w-8 text-right">
              {limit}
            </span>
          </div>

          {/* Predict Button */}
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="mt-2 w-full py-3 bg-white text-black font-bold rounded-xl shadow-[0_0_20px_rgba(255,255,255,0.3)] hover:shadow-[0_0_30px_rgba(255,255,255,0.5)] hover:scale-[1.02] active:scale-[0.98] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <span>Update Prediction</span>
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

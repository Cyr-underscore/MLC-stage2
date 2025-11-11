import { useState } from "react";
import '../style/slider.css';

interface SliderProps {
    onDecisionMade: (value: number | null) => void;
}

export default function Slider({ onDecisionMade }: SliderProps) {
    const [value, setValue] = useState<number | null>(null);
    const [unsure, setUnsure] = useState(false);

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = Number(e.target.value);
        setValue(newValue);
        setUnsure(false);
        onDecisionMade(newValue);
    };

    const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const checked = e.target.checked;
        setUnsure(checked);
        if (checked) {
            setValue(null);
            onDecisionMade(null);
        }
    };

    return (
        <div className="slider">
            <label>
                <input
                    type="checkbox"
                    checked={unsure}
                    onChange={handleCheckboxChange}
                />
                I don't know
            </label>

            {!unsure && (
                <div className="slider-container">
                    <input 
                        type="range"
                        min={0}
                        max={100}
                        step={10}
                        value={value ?? 0}
                        onChange={handleSliderChange}
                        style={{
                          background: `linear-gradient(to right, var(--main) ${value ?? 0}%, #ccc ${value ?? 0}%)`
                        }}
                    />
                    <p>{value !== null ? `${value}%` : '0%'}</p>
                </div>
            )}
        </div>
    );
}


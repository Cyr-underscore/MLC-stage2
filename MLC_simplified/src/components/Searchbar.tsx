import { useState } from "react";
import searchIcon from "../assets/search.png";
import "../style/Searchbar.css";

type Props = {
  searchTitle: string;
  onSearch?: (value: string) => void;
};

export default function SearchBar({ searchTitle, onSearch }: Props) {
  const [value, setValue] = useState("");

  const variantClass = `${searchTitle.replace(/\s+/g, '-')}`;

  return (
    <div className={`search-input-containter-${variantClass}`}>
      <div className="search-title">
        <p>{searchTitle}</p>
      </div>

      <div className="search-input">
        <input
          type="text"
          value={value}
          onChange={(e) => {
            const newValue = e.target.value;
            setValue(newValue);
            onSearch?.(newValue);
          }}
          placeholder="..."
          className={`search-input-field-${variantClass}`}
        />
        <img src={searchIcon} alt="Search Icon" className="search-icon" />
      </div>
    </div>
  );
}

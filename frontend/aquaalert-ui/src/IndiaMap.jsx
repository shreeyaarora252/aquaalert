import React, { useEffect, useRef, useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

const COLOR_MAP = {
  red: "#dc2626",
  yellow: "#ca8a04",
  green: "#16a34a",
};

const NEUTRAL_FILL = "rgba(255,255,255,0.08)";
const INDIA_SVG_URL = "/india.svg";

const IndiaMap = ({ data }) => {
  const objectRef = useRef(null);

  const [hovered, setHovered] = useState(null);
  const [locked, setLocked] = useState(null);
  const [cursor, setCursor] = useState({ x: 0, y: 0 });

  const getStateData = (stateId) =>
    data.find(
      (d) =>
        d.state.toLowerCase().replace(/\s/g, "") ===
        stateId.toLowerCase()
    );

  useEffect(() => {
    const obj = objectRef.current;
    if (!obj) return;

    const handleLoad = () => {
      const svg = obj.contentDocument;
      if (!svg) return;

      const handleMouseMove = (e) => {
        setCursor({ x: e.clientX, y: e.clientY });
      };

      const handleMouseOver = (e) => {
        if (locked) return;

        const group = e.target.closest("g[id]");
        if (!group) return;

        const match = getStateData(group.id);
        if (!match) return;

        setHovered(match);

        group.querySelectorAll("path").forEach((p) => {
          p.style.fill = COLOR_MAP[match.color];
          p.classList.add("active");
        });
      };

      const handleMouseOut = (e) => {
        if (locked) return;

        const group = e.target.closest("g[id]");
        if (!group) return;

        group.querySelectorAll("path").forEach((p) => {
          p.style.fill = NEUTRAL_FILL;
          p.classList.remove("active");
        });

        setHovered(null);
      };

      const handleClick = (e) => {
        const group = e.target.closest("g[id]");

        // Click outside → unlock
        if (!group) {
          setLocked(null);
          setHovered(null);
          return;
        }

        const match = getStateData(group.id);
        if (!match) return;

        setLocked(match);
        setHovered(match);

        group.querySelectorAll("path").forEach((p) => {
          p.style.fill = COLOR_MAP[match.color];
          p.classList.add("active");
        });
      };

      svg.addEventListener("mousemove", handleMouseMove);
      svg.addEventListener("mouseover", handleMouseOver);
      svg.addEventListener("mouseout", handleMouseOut);
      svg.addEventListener("click", handleClick);

      return () => {
        svg.removeEventListener("mousemove", handleMouseMove);
        svg.removeEventListener("mouseover", handleMouseOver);
        svg.removeEventListener("mouseout", handleMouseOut);
        svg.removeEventListener("click", handleClick);
      };
    };

    obj.addEventListener("load", handleLoad);
    return () => obj.removeEventListener("load", handleLoad);
  }, [data, locked]);

  return (
    <div className="map-shell">
      <TransformWrapper
        minScale={1}
        maxScale={4}
        initialScale={1}
        wheel={{ step: 0.08 }}
        pinch={{ step: 5 }}
        doubleClick={{ disabled: true }}
        panning={{ velocityDisabled: true }}
      >
        <TransformComponent>
          <div style={{ position: "relative" }}>
            <object
              ref={objectRef}
              type="image/svg+xml"
              data={INDIA_SVG_URL}
              className="india-map"
            />
  
            {hovered && (
              <div
                className="tooltip"
                style={{
                  left: cursor.x + 16,
                  top: cursor.y + 16,
                }}
              >
                <strong>{hovered.state}</strong>
                <div>WSI: {hovered.predicted_wsi}</div>
                <div
                  style={{
                    marginTop: "6px",
                    color:
                      hovered.color === "red"
                        ? "#f87171"
                        : hovered.color === "yellow"
                        ? "#facc15"
                        : "#4ade80",
                  }}
                >
                  {hovered.scarcity} Scarcity
                </div>
              </div>
            )}
          </div>
        </TransformComponent>
      </TransformWrapper>
    </div>
  );
}  
export default IndiaMap;

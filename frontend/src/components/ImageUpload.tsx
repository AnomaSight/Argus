import { useEffect, useState } from "react";

// C5 accepts PNG, JPEG, BMP or TIFF up to 10 MB. Checked here too, so the user hears it early.
const TYPES = ["image/png", "image/jpeg", "image/bmp", "image/tiff"];
const MAX_BYTES = 10 * 1024 * 1024;

type Props = {
  onFile: (file: File) => void;
  onError: (message: string) => void;
  big?: boolean; // the empty-state viewfinder; small once a result is showing
  busy?: boolean; // scoring: the viewfinder runs a scan line
};

export default function ImageUpload({ onFile, onError, big = false, busy = false }: Props) {
  const [over, setOver] = useState(false);

  const take = (file: File | undefined | null) => {
    if (!file) return;
    if (!TYPES.includes(file.type)) return onError("Use a PNG, JPEG, BMP or TIFF image.");
    if (file.size > MAX_BYTES) return onError("That image is over 10 MB.");
    onFile(file);
  };

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    const paste = (e: ClipboardEvent) => {
      const item = [...(e.clipboardData?.items ?? [])].find((i) => i.kind === "file");
      if (item) take(item.getAsFile());
    };
    window.addEventListener("paste", paste);
    return () => window.removeEventListener("paste", paste);
  });

  const cls = ["drop", big && "big", over && "over", busy && "scanning"].filter(Boolean).join(" ");

  return (
    <label
      className={cls}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        take(e.dataTransfer.files[0]);
      }}
    >
      <input type="file" accept={TYPES.join(",")} onChange={(e) => take(e.target.files?.[0])} />
      {big ? (
        <>
          <span className="corner tl" aria-hidden="true" />
          <span className="corner tr" aria-hidden="true" />
          <span className="corner bl" aria-hidden="true" />
          <span className="corner br" aria-hidden="true" />
          <span className="frame-note top">224 × 224 input frame</span>
          <span className="frame-note bottom">PNG, JPEG, BMP or TIFF, up to 10 MB</span>
          <span className="drop-main">
            <strong>{busy ? "Scanning" : over ? "Let go to inspect" : "Place a part in frame"}</strong>
            <span>Choose an image, drop it here, or paste it</span>
          </span>
        </>
      ) : (
        <span>
          <strong>Try another image</strong> or drop it here
        </span>
      )}
    </label>
  );
}

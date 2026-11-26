import React from "react";

type BackgroundPictureProps = {
  src: string;
  opacity?: number;
};

export default function BackgroundPicture({
  src,
  opacity = 40,
}: BackgroundPictureProps) {
  const sizes = "100vw";
  const priority = false;
  const wrapperClassName =
    "pointer-events-none fixed inset-0 -z-10 h-full w-full overflow-hidden";
  const imgClassName =
    "absolute inset-0 h-full w-full scale-[1.02] object-cover opacity-[var(--picture-opacity)] blur-xl dark:opacity-20";
  const overlayClassName =
    "absolute inset-0 h-full w-full bg-gradient-to-b from-transparent to-white dark:from-background/40 dark:to-background";
  const loading = priority ? "eager" : "lazy";
  const fetchPriority = priority ? "high" : "auto";
  const alt = "Background";

  return (
    <picture className={wrapperClassName}>
      <img
        className={imgClassName}
        style={{ "--picture-opacity": opacity / 100 } as React.CSSProperties}
        alt={alt}
        aria-hidden={alt ? undefined : true}
        src={src}
        sizes={sizes}
        loading={loading as any}
        fetchPriority={fetchPriority as any}
      />
      <div className={overlayClassName} />
    </picture>
  );
}

/**
 * Home Page
 * Displays system overview and quick access shortcuts
 */

'use client';

import { useTranslations } from "next-intl";
import { HomeCostStats } from "@/components/home/HomeCostStats";

/**
 * Home Page Component
 */
export default function HomePage() {
  const t = useTranslations("home");

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold">{t("title")}</h1>
        <p className="mt-1 text-muted-foreground">{t("description")}</p>
      </div>

      <HomeCostStats />
    </div>
  );
}

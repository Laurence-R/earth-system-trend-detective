export type MockLocation = {
  id: string;
  nameEn: string;
  lat: number;
  lon: number;
};

export const mockLocations: MockLocation[] = [
  { id: "taichung", nameEn: "Taichung", lat: 24.15, lon: 120.68 },
  { id: "taipei", nameEn: "Taipei", lat: 25.03, lon: 121.56 },
  { id: "kaohsiung", nameEn: "Kaohsiung", lat: 22.63, lon: 120.3 },
  { id: "hualien", nameEn: "Hualien", lat: 23.99, lon: 121.6 },
  { id: "yushan", nameEn: "Yushan", lat: 23.47, lon: 120.95 },
];

export const taiwanOverview = { lat: 23.7, lon: 121.0 };

export type ParticleMotion = "fall" | "drift";

export type MockVariable = {
  id: string;
  nameEn: string;
  glassColor: string;
  particleColor: string;
  motion: ParticleMotion;
};

export const mockVariables: MockVariable[] = [
  {
    id: "imerg_precip",
    nameEn: "Precipitation",
    glassColor: "#8ec8ff",
    particleColor: "#e7f6ff",
    motion: "fall",
  },
  {
    id: "gldas_tair",
    nameEn: "Near-surface air temperature",
    glassColor: "#ffb086",
    particleColor: "#ffe0cc",
    motion: "drift",
  },
  {
    id: "gldas_soilmoist",
    nameEn: "Surface soil moisture",
    glassColor: "#6f8f5a",
    particleColor: "#d5e2b8",
    motion: "drift",
  },
  {
    id: "gldas_evap",
    nameEn: "Evapotranspiration",
    glassColor: "#7fd0c8",
    particleColor: "#d7fffb",
    motion: "drift",
  },
  {
    id: "airs_tsurf",
    nameEn: "Surface air temperature",
    glassColor: "#ff8a6a",
    particleColor: "#ffd0c2",
    motion: "drift",
  },
  {
    id: "airs_h2ovap",
    nameEn: "Total water vapor",
    glassColor: "#b7d4ff",
    particleColor: "#eef5ff",
    motion: "drift",
  },
  {
    id: "merra2_t2m",
    nameEn: "2 m air temperature",
    glassColor: "#ffd27a",
    particleColor: "#fff1cc",
    motion: "drift",
  },
  {
    id: "omi_no2",
    nameEn: "Tropospheric NO2",
    glassColor: "#c084fc",
    particleColor: "#f3e8ff",
    motion: "drift",
  },
  {
    id: "omi_o3",
    nameEn: "Total ozone",
    glassColor: "#67e8f9",
    particleColor: "#e0fbff",
    motion: "drift",
  },
  {
    id: "omi_aerosol",
    nameEn: "UV aerosol index",
    glassColor: "#d6d3d1",
    particleColor: "#f5f5f4",
    motion: "drift",
  },
  {
    id: "modis_ndvi",
    nameEn: "Vegetation greenness",
    glassColor: "#4ade80",
    particleColor: "#dcfce7",
    motion: "drift",
  },
  {
    id: "modis_lst",
    nameEn: "Land surface temperature",
    glassColor: "#fb7185",
    particleColor: "#ffe4e6",
    motion: "drift",
  },
];

export function mockNormalized(variableId: string, locationId: string): number {
  let hash = 2166136261;
  const text = `${variableId}:${locationId}`;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967296;
}

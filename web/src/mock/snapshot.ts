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

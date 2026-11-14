import { describe, expect, it } from "vitest";
import { mockLocations, mockNormalized, mockVariables } from "./snapshot";

describe("mockNormalized", () => {
  it("stays inside 0 to 1 and does not change between calls", () => {
    const first = mockNormalized("imerg_precip", "taichung");
    const second = mockNormalized("imerg_precip", "taichung");
    expect(first).toBe(second);
    expect(first).toBeGreaterThanOrEqual(0);
    expect(first).toBeLessThan(1);
  });

  it("gives every variable a value at every location", () => {
    for (const variable of mockVariables) {
      for (const location of mockLocations) {
        const value = mockNormalized(variable.id, location.id);
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThan(1);
      }
    }
  });
});
